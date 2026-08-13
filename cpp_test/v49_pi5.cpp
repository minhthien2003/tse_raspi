// v49_pi5.cpp
// C++/Raspberry Pi 5 deployment of v49_pi5.py
//
// Runtime:
//   ONNX Runtime C++
//   FFTW3
//   libsndfile
//   PortAudio
//
// Model inputs:
//   inp, spk_emb, mr, mi, mm
// Model outputs:
//   est_r, est_i
//
// Audio/model parameters are intentionally kept identical to the Python source:
//   SR=16000, N_FFT=512, HOP=128, CHUNK=3s, processing hop=1.5s.

#include <onnxruntime_cxx_api.h>
#include <fftw3.h>
#include <sndfile.h>
#include <portaudio.h>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static constexpr int SR = 16000;
static constexpr int N_FFT = 512;
static constexpr int HOP = 128;
static constexpr int N_FREQ = N_FFT / 2 + 1;
static constexpr float COMP = 0.3f;
static constexpr float EPS = 1e-8f;

static constexpr int CHUNK_SAMPLES = 3 * SR;
static constexpr int HOP_SAMPLES = static_cast<int>(1.5 * SR);

static std::vector<float> make_hann(int n) {
    std::vector<float> w(n);
    constexpr double pi = 3.14159265358979323846;
    for (int i = 0; i < n; ++i)
        w[i] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * pi * i / (n - 1)));
    return w;
}

static const std::vector<float> FFT_WIN = make_hann(N_FFT);
static const std::vector<float> FADE_WIN = make_hann(CHUNK_SAMPLES);

struct Spectrogram {
    std::vector<float> real; // [N_FREQ, frames]
    std::vector<float> imag; // [N_FREQ, frames]
    std::vector<float> mag;  // [N_FREQ, frames]
    int frames = 0;
};

static Spectrogram stft(const std::vector<float>& audio) {
    const int pad = N_FFT / 2;
    std::vector<float> x(audio.size() + 2 * pad);

    // numpy.pad(..., mode="reflect") equivalent for this signal length.
    for (int i = 0; i < pad; ++i)
        x[i] = audio[pad - i];
    std::copy(audio.begin(), audio.end(), x.begin() + pad);
    for (int i = 0; i < pad; ++i)
        x[pad + audio.size() + i] = audio[audio.size() - 2 - i];

    const int frames = 1 + static_cast<int>((x.size() - N_FFT) / HOP);
    Spectrogram s;
    s.frames = frames;
    s.real.resize(static_cast<size_t>(N_FREQ) * frames);
    s.imag.resize(static_cast<size_t>(N_FREQ) * frames);
    s.mag.resize(static_cast<size_t>(N_FREQ) * frames);

    std::vector<float> in(N_FFT);
    fftwf_complex* out = static_cast<fftwf_complex*>(fftwf_malloc(sizeof(fftwf_complex) * N_FREQ));
    fftwf_plan plan = fftwf_plan_dft_r2c_1d(N_FFT, in.data(), out, FFTW_ESTIMATE);
    if (!plan) throw std::runtime_error("FFTW plan creation failed");

    for (int t = 0; t < frames; ++t) {
        const int start = t * HOP;
        for (int k = 0; k < N_FFT; ++k)
            in[k] = x[start + k] * FFT_WIN[k];

        fftwf_execute(plan);

        for (int f = 0; f < N_FREQ; ++f) {
            float r = out[f][0];
            float im = out[f][1];
            const size_t idx = static_cast<size_t>(f) * frames + t;
            s.real[idx] = r;
            s.imag[idx] = im;
            s.mag[idx] = std::sqrt(r * r + im * im + EPS);
        }
    }

    fftwf_destroy_plan(plan);
    fftwf_free(out);
    return s;
}

// Inverse STFT matching the Python implementation.
// real/imag are [N_FREQ, frames].
static std::vector<float> istft(const std::vector<float>& real,
                                const std::vector<float>& imag,
                                int frames,
                                int length) {
    const int pad = N_FFT / 2;
    const int out_len = (frames - 1) * HOP + N_FFT;

    std::vector<float> output(out_len, 0.0f);
    std::vector<float> win_sum(out_len, 0.0f);

    std::vector<fftwf_complex> in(N_FREQ);
    std::vector<float> out(N_FFT);
    fftwf_plan plan = fftwf_plan_dft_c2r_1d(N_FFT, in.data(), out.data(), FFTW_ESTIMATE);
    if (!plan) throw std::runtime_error("FFTW inverse plan creation failed");

    for (int t = 0; t < frames; ++t) {
        for (int f = 0; f < N_FREQ; ++f) {
            const size_t idx = static_cast<size_t>(f) * frames + t;
            in[f][0] = real[idx];
            in[f][1] = imag[idx];
        }

        fftwf_execute(plan);

        const int start = t * HOP;
        for (int k = 0; k < N_FFT; ++k) {
            // FFTW inverse is unnormalized.
            float seg = (out[k] / N_FFT) * FFT_WIN[k];
            output[start + k] += seg;
            win_sum[start + k] += FFT_WIN[k] * FFT_WIN[k];
        }
    }

    fftwf_destroy_plan(plan);

    std::vector<float> result(length, 0.0f);
    for (int i = 0; i < length; ++i) {
        int p = i + pad;
        if (p < 0 || p >= out_len) continue;
        result[i] = output[p] / std::max(win_sum[p], 1e-8f);
    }
    return result;
}

static std::vector<float> normalize_input(const std::vector<float>& mag) {
    std::vector<float> comp(mag.size());
    double sum = 0.0;
    for (size_t i = 0; i < mag.size(); ++i) {
        comp[i] = std::pow(mag[i] + EPS, COMP);
        sum += comp[i];
    }
    const double mean = sum / comp.size();

    double ss = 0.0;
    for (float v : comp) {
        const double d = v - mean;
        ss += d * d;
    }
    const double stddev = std::max(std::sqrt(ss / comp.size()), 1e-3);

    for (float& v : comp)
        v = static_cast<float>((v - mean) / stddev);
    return comp;
}

static std::vector<float> load_npy_float32_1d(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open embedding: " + path);

    char magic[6];
    f.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0)
        throw std::runtime_error("Not a NumPy .npy file: " + path);

    uint8_t major = 0, minor = 0;
    f.read(reinterpret_cast<char*>(&major), 1);
    f.read(reinterpret_cast<char*>(&minor), 1);

    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t h = 0;
        f.read(reinterpret_cast<char*>(&h), 2);
        header_len = h;
    } else if (major == 2 || major == 3) {
        f.read(reinterpret_cast<char*>(&header_len), 4);
    } else {
        throw std::runtime_error("Unsupported NPY version");
    }

    std::string header(header_len, '\0');
    f.read(header.data(), header_len);

    // This deployment expects the float32 embedding produced by np.save.
    if (header.find("'descr': '<f4'") == std::string::npos &&
        header.find("\"descr\": \"<f4\"") == std::string::npos) {
        throw std::runtime_error("speaker_emb.npy is not little-endian float32");
    }

    size_t count = 1;
    auto p = header.find("'shape':");
    if (p == std::string::npos) p = header.find("\"shape\":");
    if (p != std::string::npos) {
        auto lp = header.find('(', p);
        auto rp = header.find(')', lp);
        if (lp != std::string::npos && rp != std::string::npos) {
            std::string shape = header.substr(lp + 1, rp - lp - 1);
            size_t pos = 0;
            while (pos < shape.size()) {
                while (pos < shape.size() && !std::isdigit(static_cast<unsigned char>(shape[pos]))) ++pos;
                if (pos >= shape.size()) break;
                size_t end = pos;
                while (end < shape.size() && std::isdigit(static_cast<unsigned char>(shape[end]))) ++end;
                count *= static_cast<size_t>(std::stoull(shape.substr(pos, end - pos)));
                pos = end;
            }
        }
    }

    std::vector<float> data(count);
    f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
    if (!f) throw std::runtime_error("Failed reading embedding data");
    return data;
}

class V49Extractor {
public:
    V49Extractor(const std::string& model_path,
                 const std::string& emb_path,
                 float mask_power,
                 float gain_db)
        : env_(ORT_LOGGING_LEVEL_WARNING, "V49"),
          memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
          mask_power_(mask_power),
          output_gain_(std::pow(10.0f, gain_db / 20.0f)) {

        Ort::SessionOptions opts;
        opts.SetInterOpNumThreads(4);
        opts.SetIntraOpNumThreads(4);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), opts);

        spk_emb_ = load_npy_float32_1d(emb_path);
        std::cout << "  speaker embedding: " << spk_emb_.size() << " floats\n";

        Ort::AllocatorWithDefaultOptions allocator;
        std::cout << "  model inputs:\n";
        for (size_t i = 0; i < session_->GetInputCount(); ++i) {
            auto name = session_->GetInputNameAllocated(i, allocator);
            std::cout << "    " << name.get() << "\n";
        }
        std::cout << "  model outputs:\n";
        for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
            auto name = session_->GetOutputNameAllocated(i, allocator);
            std::cout << "    " << name.get() << "\n";
        }
    }

    std::vector<float> process(const std::vector<float>& audio_chunk) {
        double ss = 0.0;
        for (float x : audio_chunk) ss += static_cast<double>(x) * x;
        const float rms = static_cast<float>(std::sqrt(ss / audio_chunk.size()));
        if (rms < 1e-6f) return std::vector<float>(audio_chunk.size(), 0.0f);

        const float gain_in = 0.1f / rms;
        std::vector<float> normalized(audio_chunk.size());
        for (size_t i = 0; i < audio_chunk.size(); ++i)
            normalized[i] = audio_chunk[i] * gain_in;

        Spectrogram s = stft(normalized);
        const int frames = s.frames;
        const size_t spec_size = static_cast<size_t>(N_FREQ) * frames;

        std::vector<float> inp = normalize_input(s.mag);

        std::array<int64_t, 3> spec_shape{1, N_FREQ, frames};
        std::array<int64_t, 2> emb_shape{1, static_cast<int64_t>(spk_emb_.size())};

        auto tensor = [&](std::vector<float>& data, const int64_t* shape, size_t rank) {
            size_t n = 1;
            for (size_t i = 0; i < rank; ++i) n *= static_cast<size_t>(shape[i]);
            return Ort::Value::CreateTensor<float>(
                memory_info_, data.data(), n, shape, rank);
        };

        std::vector<Ort::Value> inputs;
        inputs.emplace_back(tensor(inp, spec_shape.data(), 3));
        inputs.emplace_back(tensor(spk_emb_, emb_shape.data(), 2));
        inputs.emplace_back(tensor(s.real, spec_shape.data(), 3));
        inputs.emplace_back(tensor(s.imag, spec_shape.data(), 3));
        inputs.emplace_back(tensor(s.mag, spec_shape.data(), 3));

        const char* input_names[] = {"inp", "spk_emb", "mr", "mi", "mm"};
        const char* output_names[] = {"est_r", "est_i"};

        auto outputs = session_->Run(
            Ort::RunOptions{nullptr},
            input_names, inputs.data(), inputs.size(),
            output_names, 2);

        float* est_r_ptr = outputs[0].GetTensorMutableData<float>();
        float* est_i_ptr = outputs[1].GetTensorMutableData<float>();

        auto shape_r = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        size_t out_count = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();

        if (out_count != spec_size) {
            throw std::runtime_error(
                "Unexpected est_r size: " + std::to_string(out_count) +
                ", expected " + std::to_string(spec_size));
        }

        std::vector<float> est_r(est_r_ptr, est_r_ptr + out_count);
        std::vector<float> est_i(est_i_ptr, est_i_ptr + out_count);

        if (mask_power_ != 1.0f) {
            for (size_t i = 0; i < spec_size; ++i) {
                float mag = std::sqrt(est_r[i] * est_r[i] + est_i[i] * est_i[i] + EPS);
                float mask = std::clamp(mag / (s.mag[i] + EPS), 0.0f, 1.0f);

                // Layout [freq, time], matching Python's [1,freq,time].
                const int f = static_cast<int>(i / frames);
                float power;
                if (f < 40) power = mask_power_ + 1.0f;
                else if (f < 80) power = mask_power_ + 0.5f;
                else power = mask_power_;

                float sharp = std::pow(mask, power);
                est_r[i] = s.real[i] * sharp;
                est_i[i] = s.imag[i] * sharp;
            }
        }

        std::vector<float> extracted = istft(est_r, est_i, frames, static_cast<int>(audio_chunk.size()));

        for (float& x : extracted)
            x = x / gain_in * output_gain_;

        return extracted;
    }

private:
    Ort::Env env_;
    Ort::MemoryInfo memory_info_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<float> spk_emb_;
    float mask_power_;
    float output_gain_;
};

static std::vector<float> read_wav_mono(const std::string& path, int& sr) {
    SF_INFO info{};
    SNDFILE* sf = sf_open(path.c_str(), SFM_READ, &info);
    if (!sf) throw std::runtime_error("Cannot open WAV: " + path);

    std::vector<float> interleaved(static_cast<size_t>(info.frames) * info.channels);
    sf_readf_float(sf, interleaved.data(), info.frames);
    sf_close(sf);

    sr = info.samplerate;
    std::vector<float> mono(static_cast<size_t>(info.frames));
    for (sf_count_t i = 0; i < info.frames; ++i) {
        double s = 0.0;
        for (int c = 0; c < info.channels; ++c)
            s += interleaved[static_cast<size_t>(i) * info.channels + c];
        mono[static_cast<size_t>(i)] = static_cast<float>(s / info.channels);
    }
    return mono;
}

static void write_wav_mono(const std::string& path, const std::vector<float>& audio) {
    SF_INFO info{};
    info.samplerate = SR;
    info.channels = 1;
    info.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

    SNDFILE* sf = sf_open(path.c_str(), SFM_WRITE, &info);
    if (!sf) throw std::runtime_error("Cannot write WAV: " + path);

    std::vector<short> pcm(audio.size());
    for (size_t i = 0; i < audio.size(); ++i) {
        float x = std::clamp(audio[i], -1.0f, 1.0f);
        pcm[i] = static_cast<short>(std::lrint(x * 32767.0f));
    }
    sf_writef_short(sf, pcm.data(), static_cast<sf_count_t>(pcm.size()));
    sf_close(sf);
}

static void process_file(V49Extractor& extractor,
                         const std::string& input,
                         const std::string& output) {
    int sr = 0;
    std::vector<float> audio = read_wav_mono(input, sr);

    if (sr != SR)
        std::cerr << "WARNING: file is " << sr << " Hz, model expects " << SR << " Hz\n";
    if (audio.size() < CHUNK_SAMPLES)
        throw std::runtime_error("Input file must be at least 3 seconds for this implementation");

    std::vector<float> out(audio.size(), 0.0f);
    std::vector<float> weight(audio.size(), 0.0f);

    int n = 0;
    double total_ms = 0.0;

    for (size_t start = 0; start + CHUNK_SAMPLES <= audio.size(); start += HOP_SAMPLES) {
        std::vector<float> chunk(audio.begin() + start, audio.begin() + start + CHUNK_SAMPLES);

        auto t0 = std::chrono::steady_clock::now();
        std::vector<float> y = extractor.process(chunk);
        auto t1 = std::chrono::steady_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;

        for (int i = 0; i < CHUNK_SAMPLES; ++i) {
            out[start + i] += y[i] * FADE_WIN[i];
            weight[start + i] += FADE_WIN[i];
        }
        ++n;
    }

    for (size_t i = 0; i < out.size(); ++i)
        out[i] = weight[i] > 1e-8f ? out[i] / weight[i] : 0.0f;

    write_wav_mono(output, out);

    const double avg = total_ms / std::max(n, 1);
    const double rtf = avg / (CHUNK_SAMPLES * 1000.0 / SR);

    std::cout << "  " << input << " -> " << output << "\n";
    std::cout << "  " << n << " chunks, avg "
              << std::fixed << std::setprecision(0) << avg
              << " ms, RTF " << std::setprecision(3) << rtf << "\n";
}

static double benchmark(V49Extractor& extractor, int n_iters = 20) {
    std::mt19937 rng(1234);
    std::normal_distribution<float> d(0.0f, 0.01f);
    std::vector<float> dummy(CHUNK_SAMPLES);
    for (auto& x : dummy) x = d(rng);

    for (int i = 0; i < 3; ++i) extractor.process(dummy);

    std::vector<double> times;
    times.reserve(n_iters);

    for (int i = 0; i < n_iters; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        extractor.process(dummy);
        auto t1 = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    std::sort(times.begin(), times.end());
    double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    double median = times[times.size() / 2];
    double rtf = mean / 3000.0;

    std::cout << "\nBenchmark (" << n_iters << " iterations, 3s chunks)\n";
    std::cout << "  mean:   " << mean << " ms\n";
    std::cout << "  median: " << median << " ms\n";
    std::cout << "  min:    " << times.front() << " ms\n";
    std::cout << "  max:    " << times.back() << " ms\n";
    std::cout << "  RTF:    " << rtf << (rtf < 1.0 ? " (real-time OK)" : " (TOO SLOW)") << "\n";
    return rtf;
}

struct RealtimeContext {
    V49Extractor* extractor = nullptr;
    std::vector<float> input_buffer = std::vector<float>(CHUNK_SAMPLES, 0.0f);
    std::vector<float> output_buffer = std::vector<float>(CHUNK_SAMPLES, 0.0f);
    std::vector<float> output_weight = std::vector<float>(CHUNK_SAMPLES, 0.0f);

    int samples_received = 0;
    std::atomic<bool> running{true};
    std::atomic<long> chunks_processed{0};
    double total_latency_ms = 0.0;
    //std::atomic<double> total_latency_ms{0.0};

    std::vector<float> rec_input;
    std::vector<float> rec_output;
    std::string save_prefix;

    //std::mutex process_mutex;
};

static int pa_callback(const void* input,
                       void* output,
                       unsigned long frames,
                       const PaStreamCallbackTimeInfo*,
                       PaStreamCallbackFlags,
                       void* userData) {
    auto* ctx = static_cast<RealtimeContext*>(userData);
    const float* in = static_cast<const float*>(input);
    float* out = static_cast<float*>(output);

    const int shift = static_cast<int>(frames);

    for (int i = 0; i < shift; ++i) {
        float x = in ? in[i] : 0.0f;
        ctx->rec_input.push_back(x);
        std::rotate(ctx->input_buffer.begin(),
                    ctx->input_buffer.begin() + 1,
                    ctx->input_buffer.end());
        ctx->input_buffer.back() = x;
    }

    ctx->samples_received += shift;

    if (ctx->samples_received >= HOP_SAMPLES) {
        ctx->samples_received -= HOP_SAMPLES;

        auto t0 = std::chrono::steady_clock::now();
        std::vector<float> extracted = ctx->extractor->process(ctx->input_buffer);
        auto t1 = std::chrono::steady_clock::now();

        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        //ctx->total_latency_ms.fetch_add(ms);
        ctx->total_latency_ms += ms;
        ctx->chunks_processed.fetch_add(1);

        for (int i = 0; i < CHUNK_SAMPLES; ++i) {
            ctx->output_buffer[i] += extracted[i] * FADE_WIN[i];
            ctx->output_weight[i] += FADE_WIN[i];
        }
    }

    for (int i = 0; i < shift; ++i) {
        float w = ctx->output_weight[i];
        float y = w > 1e-8f ? ctx->output_buffer[i] / w : 0.0f;
        out[i] = y;
        ctx->rec_output.push_back(y);
    }

    std::rotate(ctx->output_buffer.begin(),
                ctx->output_buffer.begin() + shift,
                ctx->output_buffer.end());
    std::fill(ctx->output_buffer.end() - shift, ctx->output_buffer.end(), 0.0f);

    std::rotate(ctx->output_weight.begin(),
                ctx->output_weight.begin() + shift,
                ctx->output_weight.end());
    std::fill(ctx->output_weight.end() - shift, ctx->output_weight.end(), 0.0f);

    return paContinue;
}

// PortAudio tham do het backend ALSA/JACK/OSS luc khoi tao, moi backend hong
// deu in loi ra stderr ("jack server is not running", "Cannot open /dev/dsp",
// "Unknown PCM iec958..."). Deu vo hai. Nuot stderr trong luc do; loi that van
// duoc bao qua ma loi PaError nen khong mat thong tin.
class StderrSilencer {
public:
    explicit StderrSilencer(bool active) {
        if (!active) return;
        std::fflush(stderr);
        saved_ = dup(STDERR_FILENO);
        int devnull = open("/dev/null", O_WRONLY);
        if (saved_ >= 0 && devnull >= 0) dup2(devnull, STDERR_FILENO);
        if (devnull >= 0) close(devnull);
    }
    ~StderrSilencer() { restore(); }
    void restore() {
        if (saved_ < 0) return;
        std::fflush(stderr);
        dup2(saved_, STDERR_FILENO);
        close(saved_);
        saved_ = -1;
    }
    StderrSilencer(const StderrSilencer&) = delete;
    StderrSilencer& operator=(const StderrSilencer&) = delete;
private:
    int saved_ = -1;
};

// spec rong  -> thiet bi mac dinh
// spec la so -> chi so thiet bi
// nguoc lai  -> khop chuoi con, khong phan biet hoa thuong, trong ten thiet bi
static PaDeviceIndex resolve_device(const std::string& spec, bool input) {
    if (spec.empty())
        return input ? Pa_GetDefaultInputDevice() : Pa_GetDefaultOutputDevice();

    bool numeric = !spec.empty() &&
                   spec.find_first_not_of("0123456789") == std::string::npos;
    if (numeric) {
        int idx = std::stoi(spec);
        if (idx < 0 || idx >= Pa_GetDeviceCount())
            throw std::runtime_error("Device index out of range: " + spec);
        return idx;
    }

    std::string needle = spec;
    std::transform(needle.begin(), needle.end(), needle.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    for (PaDeviceIndex i = 0; i < Pa_GetDeviceCount(); ++i) {
        const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
        if (!info || !info->name) continue;
        int channels = input ? info->maxInputChannels : info->maxOutputChannels;
        if (channels < 1) continue;

        std::string name = info->name;
        std::transform(name.begin(), name.end(), name.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        if (name.find(needle) != std::string::npos) return i;
    }
    throw std::runtime_error("No " + std::string(input ? "input" : "output") +
                             " device matching: " + spec);
}

static void describe_device(const char* label, PaDeviceIndex idx) {
    const PaDeviceInfo* info = Pa_GetDeviceInfo(idx);
    std::cout << "  " << label << ": [" << idx << "] "
              << (info && info->name ? info->name : "?") << "\n";
}

static void list_devices(bool quiet_probe) {
    {
        StderrSilencer hush(quiet_probe);
        PaError err = Pa_Initialize();
        hush.restore();
        if (err != paNoError) throw std::runtime_error(Pa_GetErrorText(err));
    }

    std::cout << "\nAudio devices (in/out channels):\n";
    PaDeviceIndex def_in = Pa_GetDefaultInputDevice();
    PaDeviceIndex def_out = Pa_GetDefaultOutputDevice();

    for (PaDeviceIndex i = 0; i < Pa_GetDeviceCount(); ++i) {
        const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
        if (!info) continue;
        std::cout << "  [" << std::setw(2) << i << "] "
                  << std::setw(2) << info->maxInputChannels << "in "
                  << std::setw(2) << info->maxOutputChannels << "out  "
                  << (info->name ? info->name : "?");
        if (i == def_in) std::cout << "   (default in)";
        if (i == def_out) std::cout << "   (default out)";
        std::cout << "\n";
    }

    if (def_in == paNoDevice)  std::cout << "\n  WARNING: no default input device\n";
    if (def_out == paNoDevice) std::cout << "  WARNING: no default output device\n";
    std::cout << "\nUse --in-device / --out-device with an index or a name substring.\n";

    Pa_Terminate();
}

static void realtime(V49Extractor& extractor, const std::string& save_prefix,
                     const std::string& in_spec, const std::string& out_spec,
                     bool quiet_probe) {
    RealtimeContext ctx;
    ctx.extractor = &extractor;
    ctx.save_prefix = save_prefix;

    StderrSilencer hush(quiet_probe);
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        hush.restore();
        throw std::runtime_error(Pa_GetErrorText(err));
    }

    PaStream* stream = nullptr;
    PaStreamParameters inputParams{};
    PaStreamParameters outputParams{};

    try {
        inputParams.device = resolve_device(in_spec, true);
        outputParams.device = resolve_device(out_spec, false);
    } catch (...) {
        Pa_Terminate();
        hush.restore();
        throw;
    }

    // Pa_GetDefaultInputDevice() tra paNoDevice khi khong co mic (Pi 5 khong co
    // mic tich hop). Phai chan o day, neu khong Pa_GetDeviceInfo(-1) tra nullptr
    // va deref se segfault.
    auto require = [&](PaDeviceIndex idx, const char* what, const char* flag)
                       -> const PaDeviceInfo* {
        const PaDeviceInfo* info =
            (idx == paNoDevice) ? nullptr : Pa_GetDeviceInfo(idx);
        if (!info) {
            Pa_Terminate();
            hush.restore();
            throw std::runtime_error(
                std::string("No usable ") + what + " device. "
                "Run --list-devices, then pick one with " + flag +
                " (index or name substring).");
        }
        return info;
    };

    const PaDeviceInfo* in_info = require(inputParams.device, "input", "--in-device");
    const PaDeviceInfo* out_info = require(outputParams.device, "output", "--out-device");

    inputParams.channelCount = 1;
    inputParams.sampleFormat = paFloat32;
    inputParams.suggestedLatency = in_info->defaultLowInputLatency;

    outputParams.channelCount = 1;
    outputParams.sampleFormat = paFloat32;
    outputParams.suggestedLatency = out_info->defaultLowOutputLatency;

    const unsigned long blocksize = HOP_SAMPLES;

    err = Pa_OpenStream(&stream, &inputParams, &outputParams,
                        SR, blocksize, paNoFlag, pa_callback, &ctx);
    if (err != paNoError) {
        Pa_Terminate();
        hush.restore();
        throw std::runtime_error(std::string(Pa_GetErrorText(err)) +
                                 " (try --list-devices)");
    }

    err = Pa_StartStream(stream);
    if (err != paNoError) {
        Pa_CloseStream(stream);
        Pa_Terminate();
        hush.restore();
        throw std::runtime_error(Pa_GetErrorText(err));
    }

    hush.restore();

    describe_device("input ", inputParams.device);
    describe_device("output", outputParams.device);
    std::cout << "\n  chunk: 3s, hop: 1.5s\n";
    if (!save_prefix.empty())
        std::cout << "  recording to: " << save_prefix << "_input.wav / "
                  << save_prefix << "_output.wav\n";
    std::cout << "  Ctrl+C to stop\n\n";

    while (Pa_IsStreamActive(stream) == 1) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        long n = ctx.chunks_processed.load();
        if (n > 0) {
            //double avg = ctx.total_latency_ms.load() / n;
            double avg = ctx.total_latency_ms / n;
            double rtf = avg / 3000.0;
            std::cout << "\r  chunks " << n
                      << "  latency " << std::fixed << std::setprecision(0) << avg << " ms"
                      << "  RTF " << std::setprecision(3) << rtf << std::flush;
        }
    }

    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();

    if (!save_prefix.empty() && !ctx.rec_input.empty()) {
        write_wav_mono(save_prefix + "_input.wav", ctx.rec_input);
        write_wav_mono(save_prefix + "_output.wav", ctx.rec_output);
        std::cout << "\nSaved recordings.\n";
    }
}

static void usage(const char* p) {
    std::cout <<
        "Usage:\n"
        "  " << p << " --model v49_int8.onnx --emb speaker_emb.npy [options]\n\n"
        "Options:\n"
        "  --model PATH       ONNX model\n"
        "  --emb PATH         speaker_emb.npy\n"
        "  --power N          mask sharpening (1=off, 2=default, 3=aggressive)\n"
        "  --gain N           output gain in dB\n"
        "  --file INPUT.wav   process WAV instead of live audio\n"
        "  -o OUTPUT.wav      output WAV in file mode\n"
        "  --save PREFIX      save realtime input/output WAV\n"
        "  --bench            benchmark inference and exit\n"
        "  --list-devices     list audio devices and exit\n"
        "  --in-device SPEC   input device: index or name substring\n"
        "  --out-device SPEC  output device: index or name substring\n"
        "  --verbose-audio    show ALSA/JACK probe messages on stderr\n";
}

int main(int argc, char** argv) {
    try {
        std::string model, emb, file, output = "extracted.wav", save;
        std::string in_device, out_device;
        float power = 2.0f, gain = 2.0f;
        bool bench = false, list_dev = false, quiet_probe = true;

        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto need = [&](const char* name) -> std::string {
                if (i + 1 >= argc) throw std::runtime_error(std::string("Missing value for ") + name);
                return argv[++i];
            };

            if (a == "--model") model = need("--model");
            else if (a == "--emb") emb = need("--emb");
            else if (a == "--power") power = std::stof(need("--power"));
            else if (a == "--gain") gain = std::stof(need("--gain"));
            else if (a == "--file") file = need("--file");
            else if (a == "-o" || a == "--output") output = need("--output");
            else if (a == "--save") save = need("--save");
            else if (a == "--bench") bench = true;
            else if (a == "--list-devices") list_dev = true;
            else if (a == "--in-device") in_device = need("--in-device");
            else if (a == "--out-device") out_device = need("--out-device");
            else if (a == "--verbose-audio") quiet_probe = false;
            else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
            else throw std::runtime_error("Unknown argument: " + a);
        }

        // Liet ke thiet bi khong can model, chay duoc truoc khi cau hinh xong.
        if (list_dev) {
            list_devices(quiet_probe);
            return 0;
        }

        if (model.empty() || emb.empty()) {
            usage(argv[0]);
            return 1;
        }

        std::cout << "==================================================\n";
        std::cout << "V49 SPEAKER EXTRACTION - C++ / Raspberry Pi 5\n";
        std::cout << "  model: " << model << "\n";
        std::cout << "  power: " << power << "  gain: +" << gain << " dB\n";
        std::cout << "==================================================\n";

        V49Extractor extractor(model, emb, power, gain);

        if (bench) benchmark(extractor);
        else if (!file.empty()) process_file(extractor, file, output);
        else realtime(extractor, save, in_device, out_device, quiet_probe);

    } catch (const Ort::Exception& e) {
        std::cerr << "\nONNX Runtime error: " << e.what() << "\n";
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "\nERROR: " << e.what() << "\n";
        return 1;
    }
}
