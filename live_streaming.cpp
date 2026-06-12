// live_streaming.cpp
// Ultra-Low-Latency MediaMTX RTSP Forwarder
// All config lives in .env file — no hardcoded values
// Compile Windows: g++ -std=c++17 -o live_streaming.exe live_streaming.cpp -lws2_32 -lpthread
// Compile Linux:   g++ -std=c++17 -o live_streaming live_streaming.cpp -lpthread
#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <thread>
#include <atomic>
#include <chrono>
#include <sstream>
#include <algorithm>
#include <csignal>
#include <ctime>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#define popen  _popen
#define pclose _pclose
#define SOCKET_T SOCKET
#define CLOSE_SOCKET(s) closesocket(s)
#else
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
#define SOCKET_T int
#define CLOSE_SOCKET(s) close(s)
#define INVALID_SOCKET -1
#endif

// ─────────────────────────────────────────────
//  .env loader
// ─────────────────────────────────────────────
class EnvLoader {
public:
    explicit EnvLoader(const std::string& path = ".env") {
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "[ERROR] Cannot open .env file at: " << path << "\n";
            std::cerr << "[ERROR] Please create a .env file next to this program.\n";
            return;
        }
        loaded_ = true;
        std::string line;
        while (std::getline(file, line)) {
            // Strip comments
            auto comment = line.find('#');
            if (comment != std::string::npos)
                line = line.substr(0, comment);
            // Trim
            auto l = line.find_first_not_of(" \t\r\n");
            if (l == std::string::npos) continue;
            line = line.substr(l);
            auto r = line.find_last_not_of(" \t\r\n");
            if (r != std::string::npos) line = line.substr(0, r + 1);
            if (line.empty()) continue;
            // Split key=value
            auto eq = line.find('=');
            if (eq == std::string::npos) continue;
            std::string key = line.substr(0, eq);
            std::string val = line.substr(eq + 1);
            // Strip quotes
            if (val.size() >= 2 &&
                ((val.front() == '"' && val.back() == '"') ||
                    (val.front() == '\'' && val.back() == '\'')))
                val = val.substr(1, val.size() - 2);
            // Trim key
            auto kt = key.find_last_not_of(" \t");
            if (kt != std::string::npos) key = key.substr(0, kt + 1);
            // Trim value
            auto vt = val.find_first_not_of(" \t");
            if (vt != std::string::npos) val = val.substr(vt);
            store_[key] = val;
        }
    }

    // get(key) — returns value or empty string
    std::string get(const std::string& key) const {
        auto it = store_.find(key);
        if (it != store_.end() && !it->second.empty())
            return it->second;
        // Also check OS environment
        const char* ev = std::getenv(key.c_str());
        if (ev && std::string(ev).size() > 0)
            return std::string(ev);
        return "";
    }

    bool has(const std::string& key) const {
        return !get(key).empty();
    }

    bool isLoaded() const { return loaded_; }

private:
    std::map<std::string, std::string> store_;
    bool loaded_ = false;
};

// ─────────────────────────────────────────────
//  Logger
// ─────────────────────────────────────────────
enum class LogLevel { INFO, WARNING, ERR };

void log(LogLevel level, const std::string& msg) {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    char tbuf[32];
    std::strftime(tbuf, sizeof(tbuf), "%Y-%m-%d %H:%M:%S", std::localtime(&time));
    std::string prefix;
    switch (level) {
    case LogLevel::INFO:    prefix = "INFO   "; break;
    case LogLevel::WARNING: prefix = "WARNING"; break;
    case LogLevel::ERR:     prefix = "ERROR  "; break;
    }
    std::cout << tbuf << " - " << prefix << " - " << msg << "\n";
}

void logInfo(const std::string& m) { log(LogLevel::INFO, m); }
void logWarn(const std::string& m) { log(LogLevel::WARNING, m); }
void logError(const std::string& m) { log(LogLevel::ERR, m); }

// ─────────────────────────────────────────────
//  Quality presets
// ─────────────────────────────────────────────
struct QualityPreset {
    std::string scale;
    std::string video_bitrate;
    std::string max_bitrate;
    std::string buf_size;
    std::string preset;
    std::string profile;
    std::string crf;
    bool        source_copy = false;
};

const std::map<std::string, QualityPreset> PRESETS = {
    { "LOW",    { "640:360",   "1200k", "1200k", "600k",  "veryfast", "main", "28", false } },
    { "MEDIUM", { "1280:720",  "2500k", "2800k", "1250k", "veryfast", "main", "23", false } },
    { "HIGH",   { "1920:1080", "4500k", "5000k", "2250k", "faster",   "high", "18", false } },
    { "SOURCE", { "",          "",      "",      "",      "",         "",     "",   true  } },
};

// ─────────────────────────────────────────────
//  Signal handling
// ─────────────────────────────────────────────
std::atomic<bool> g_stop{ false };

void signalHandler(int) {
    std::cout << "\n[STOP] Signal received - shutting down...\n";
    g_stop = true;
}

// ─────────────────────────────────────────────
//  RTSPStreamer
// ─────────────────────────────────────────────
class RTSPStreamer {
public:
    RTSPStreamer(const std::string& cameraUrl,
        const std::string& server,
        const std::string& streamName,
        const std::string& username,
        const std::string& password,
        const std::string& quality)
        : cameraUrl_(cameraUrl)
        , server_(server)
        , streamName_(streamName)
        , quality_(quality)
    {
        // Normalize quality
        std::transform(quality_.begin(), quality_.end(), quality_.begin(), ::toupper);
        if (PRESETS.find(quality_) == PRESETS.end()) {
            logWarn("Unknown QUALITY '" + quality_ + "' - defaulting to SOURCE");
            quality_ = "SOURCE";
        }
        preset_ = PRESETS.at(quality_);
        outputUrl_ = "rtsp://" + username + ":" + password
            + "@" + server_ + ":8554/" + streamName_;

#ifdef _WIN32
        WSADATA wsa;
        WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
    }

    ~RTSPStreamer() {
#ifdef _WIN32
        WSACleanup();
#endif
    }

    // ── server reachability ─────────────────────────────────────────────
    bool checkServerConnection() const {
        struct addrinfo hints {}, * res = nullptr;
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        if (getaddrinfo(server_.c_str(), "8554", &hints, &res) != 0) {
            logError("Cannot resolve hostname: " + server_);
            return false;
        }
        SOCKET_T sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock == INVALID_SOCKET) {
            freeaddrinfo(res);
            logError("Socket creation failed");
            return false;
        }
#ifdef _WIN32
        DWORD tv = 5000;
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (char*)&tv, sizeof(tv));
        setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, (char*)&tv, sizeof(tv));
#else
        timeval tv{ 5, 0 };
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif
        bool ok = (connect(sock, res->ai_addr,
            static_cast<int>(res->ai_addrlen)) == 0);
        CLOSE_SOCKET(sock);
        freeaddrinfo(res);
        if (ok) logInfo("OK  MediaMTX reachable at " + server_ + ":8554");
        else    logError("FAIL Cannot reach " + server_ + ":8554");
        return ok;
    }

    // ── camera probe ────────────────────────────────────────────────────
    bool checkCameraConnection() const {
        logInfo("Probing camera stream...");
        std::string cmd =
            "ffprobe -v error"
            " -rtsp_transport tcp"
            " -probesize 2000000"
            " -analyzeduration 2000000"
            " -show_streams -select_streams v:0"
            " -show_entries stream=width,height,r_frame_rate,codec_name,bit_rate"
            " -of default=noprint_wrappers=1"
            " \"" + cameraUrl_ + "\" 2>&1";

        FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) { logError("ffprobe launch failed"); return false; }
        std::string out;
        char buf[256];
        while (fgets(buf, sizeof(buf), pipe)) out += buf;
        int rc = pclose(pipe);
        if (rc == 0) { logInfo("OK  Camera info:\n" + out); return true; }
        logError("FAIL Camera:\n" + out);
        return false;
    }

    // ── build FFmpeg command ─────────────────────────────────────────────
    std::string buildFFmpegCmd() const {
        std::ostringstream cmd;

        cmd << "ffmpeg -loglevel warning";

        if (preset_.source_copy) {
            // Zero re-encode — bit-perfect copy
            cmd << " -fflags nobuffer"
                << " -flags low_delay"
                << " -rtsp_transport tcp"
                << " -i \"" << cameraUrl_ << "\""
                << " -c:v copy"
                << " -c:a copy";
        }
        else {
            cmd << " -fflags nobuffer"
                << " -flags low_delay"
                << " -strict experimental"
                << " -avioflags direct"
                << " -probesize 2000000"
                << " -analyzeduration 1000000"
                << " -rtsp_transport tcp"
                << " -i \"" << cameraUrl_ << "\"";

            if (!preset_.scale.empty())
                cmd << " -vf \"scale=" << preset_.scale
                << ":force_original_aspect_ratio=decrease"
                << ",pad=" << preset_.scale << ":(ow-iw)/2:(oh-ih)/2\"";

            cmd << " -c:v libx264"
                << " -preset " << preset_.preset
                << " -tune zerolatency"
                << " -profile:v " << preset_.profile
                << " -level 4.1"
                << " -crf " << preset_.crf
                << " -maxrate " << preset_.max_bitrate
                << " -bufsize " << preset_.buf_size
                << " -g 30 -keyint_min 30 -sc_threshold 0"
                << " -colorspace bt709 -color_trc bt709 -color_primaries bt709"
                << " -threads 0"
                << " -c:a aac -b:a 128k -ar 44100 -ac 2";
        }

        cmd << " -f rtsp"
            << " -rtsp_transport tcp"
            << " -flush_packets 1"
            << " \"" << outputUrl_ << "\""
            << " 2>&1";

        return cmd.str();
    }

    // ── stderr monitor (runs in background thread) ───────────────────────
    void monitorOutput(FILE* pipe) {
        char buf[512];
        auto lastLog = std::chrono::steady_clock::now();
        while (fgets(buf, sizeof(buf), pipe) && !g_stop) {
            std::string line(buf);
            line.erase(std::remove(line.begin(), line.end(), '\n'), line.end());
            line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
            if (line.empty()) continue;
            std::string low = line;
            std::transform(low.begin(), low.end(), low.begin(), ::tolower);
            auto now = std::chrono::steady_clock::now();
            if (low.find("error") != std::string::npos ||
                low.find("invalid") != std::string::npos ||
                low.find("failed") != std::string::npos)
                logError("[FFmpeg] " + line);
            else if (low.find("warning") != std::string::npos)
                logWarn("[FFmpeg] " + line);
            else if (low.find("frame=") != std::string::npos) {
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    now - lastLog).count();
                if (elapsed >= 15) {
                    logInfo("[Stream] " + line);
                    lastLog = now;
                }
            }
        }
    }

    // ── single FFmpeg run ────────────────────────────────────────────────
    int runOnce() {
        std::string cmd = buildFFmpegCmd();
        logInfo("[START] Quality=" + quality_
            + " rtsp://" + server_ + ":8554/" + streamName_);
        printViewURLs();
        FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) { logError("Failed to launch FFmpeg"); return -1; }
        std::thread mon([this, pipe]() { monitorOutput(pipe); });
        mon.detach();
        int rc = pclose(pipe);
        return rc;
    }

    // ── main streaming loop with auto-restart ────────────────────────────
    bool startStreaming() {
        logInfo("Running pre-flight checks...");
        if (!checkServerConnection()) {
            logError("Aborting - server unreachable.");
            return false;
        }
        if (!checkCameraConnection()) {
            logError("Aborting - camera inaccessible.");
            return false;
        }
        logInfo("All checks passed.\n");

        int attempt = 0;
        while (!g_stop) {
            ++attempt;
            logInfo("[Attempt " + std::to_string(attempt) + "/"
                + std::to_string(MAX_RESTARTS) + "] Starting FFmpeg...");
            int rc = runOnce();
            if (g_stop) { logInfo("Stopped by user."); break; }
            if (rc == 0) { logInfo("FFmpeg finished cleanly."); break; }
            logError("FFmpeg exited with code " + std::to_string(rc));
            if (attempt >= MAX_RESTARTS) {
                logError("Max restarts reached. Giving up.");
                return false;
            }
            logInfo("Restarting in " + std::to_string(RESTART_DELAY) + "s...");
            for (int i = 0; i < RESTART_DELAY && !g_stop; ++i)
                std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        return true;
    }

private:
    static const int MAX_RESTARTS = 10;
    static const int RESTART_DELAY = 3;

    std::string   cameraUrl_;
    std::string   server_;
    std::string   streamName_;
    std::string   quality_;
    std::string   outputUrl_;
    QualityPreset preset_;

    void printViewURLs() const {
        std::string sep(60, '=');
        std::cout << sep << "\n"
            << "Watch your stream:\n"
            << "  WebRTC : http://" << server_ << ":8889/" << streamName_ << "\n"
            << "  HLS    : http://" << server_ << ":8888/" << streamName_ << "\n"
            << "  RTSP   : rtsp://" << server_ << ":8554/" << streamName_ << "\n"
            << sep << "\n";
    }
};

// ─────────────────────────────────────────────
//  Validate required keys exist in .env
// ─────────────────────────────────────────────
bool validateConfig(const EnvLoader& env) {
    std::vector<std::string> required = {
        "RTSP_CAMERA_URL",
        "EC2_SERVER",
        "STREAM_NAME"
    };
    std::vector<std::string> missing;
    for (const auto& key : required)
        if (!env.has(key)) missing.push_back(key);

    if (!missing.empty()) {
        logError("Missing required keys in .env:");
        for (const auto& k : missing)
            logError("  - " + k);
        logError("");
        logError("Your .env file should look like this:");
        logError("  RTSP_CAMERA_URL=rtsp://admin:admin@192.168.1.2:554/stream");
        logError("  EC2_SERVER=ec2-xx-xx-xx-xx.compute.amazonaws.com");
        logError("  STREAM_NAME=mycamera");
        logError("  MEDIAMTX_USERNAME=publisher");
        logError("  MEDIAMTX_PASSWORD=yourSecurePassword123");
        logError("  QUALITY=SOURCE");
        return false;
    }
    return true;
}

// ─────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────
int main() {
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    std::string sep(60, '=');
    std::cout << sep << "\n"
        << "  MediaMTX High-Quality Low-Latency RTSP Forwarder\n"
        << sep << "\n\n";

    // Load .env or ENV_FILE override
    const char* envFilePath = std::getenv("ENV_FILE");
    std::string envPath = envFilePath && envFilePath[0] ? envFilePath : ".env";
    EnvLoader env(envPath);
    if (!env.isLoaded()) {
        logError(envPath + " file not found next to the program.");
        logError("Create a .env file with these keys:");
        logError("  RTSP_CAMERA_URL=rtsp://...");
        logError("  EC2_SERVER=...");
        logError("  STREAM_NAME=mycamera");
        logError("  MEDIAMTX_USERNAME=publisher");
        logError("  MEDIAMTX_PASSWORD=yourpassword");
        logError("  QUALITY=SOURCE");
        return 1;
    }

    if (!validateConfig(env)) return 1;

    // Read all values from .env
    std::string camera = env.get("RTSP_CAMERA_URL");
    std::string server = env.get("EC2_SERVER");
    std::string stream = env.get("STREAM_NAME");
    std::string username = env.get("MEDIAMTX_USERNAME");
    std::string password = env.get("MEDIAMTX_PASSWORD");
    std::string quality = env.get("QUALITY");

    // Optional fields get sensible defaults if not set
    if (username.empty()) username = "publisher";
    if (password.empty()) password = "yourSecurePassword123";
    if (quality.empty())  quality = "SOURCE";

    logInfo("Config loaded from .env:");
    logInfo("  Camera   : " + camera.substr(0, 50) + "...");
    logInfo("  Server   : " + server);
    logInfo("  Stream   : " + stream);
    logInfo("  Username : " + username);
    logInfo("  Quality  : " + quality);
    std::cout << "\n";

    RTSPStreamer streamer(camera, server, stream, username, password, quality);
    return streamer.startStreaming() ? 0 : 1;
}