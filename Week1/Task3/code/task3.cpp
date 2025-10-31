#include <vector>
#include <random>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>
#include <iomanip>
using namespace std;

struct Point {
    double x, y;
    Point(double x = 0, double y = 0) : x(x), y(y) {}
};

class Simulator {
private:
    // ÔË¶¯²ÎÊý
    Point initial_position;
    Point initial_velocity;
    double delta_t;

    // ÔëÉù²ÎÊý
    double measurement_noise_std;
    double process_noise_std;

    // Ëæ»úÊýÉú³ÉÆ÷
    std::default_random_engine generator;
    std::normal_distribution<double> measurement_noise;
    std::normal_distribution<double> process_noise;

public:
    Simulator(double initial_x = 0, double initial_y = 0,
        double velocity_x = 2, double velocity_y = 3,
        double delta_time = 0.01,
        double meas_noise_std = 0.5,
        double proc_noise_std = 0.1)
        : initial_position(initial_x, initial_y)
        , initial_velocity(velocity_x, velocity_y)
        , delta_t(delta_time)
        , measurement_noise_std(meas_noise_std)
        , process_noise_std(proc_noise_std)
        , measurement_noise(0, meas_noise_std)
        , process_noise(0, proc_noise_std)
    {
        // Ê¹ÓÃÊ±¼ä×÷ÎªËæ»úÖÖ×Ó
        generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
    }

    // 1. Ä£ÄâÀíÏëºã¶¨ËÙ¶ÈÔË¶¯
    std::vector<Point> simulateIdealMotion(double total_time) {
        std::vector<Point> positions;
        int steps = static_cast<int>(total_time / delta_t);

        Point current_pos = initial_position;
        Point current_vel = initial_velocity;

        for (int i = 0; i <= steps; ++i) {
            positions.push_back(current_pos);

            // ºã¶¨ËÙ¶ÈÔË¶¯Ä£ÐÍ
            current_pos.x += current_vel.x * delta_t;
            current_pos.y += current_vel.y * delta_t;
        }

        return positions;
    }

    // 2. Ìí¼Ó²âÁ¿ÔëÉù
    std::vector<Point> addMeasurementNoise(const std::vector<Point>& true_positions) {
        std::vector<Point> observed_positions;

        for (const auto& true_pos : true_positions) {
            Point observed_pos;
            observed_pos.x = true_pos.x + measurement_noise(generator);
            observed_pos.y = true_pos.y + measurement_noise(generator);
            observed_positions.push_back(observed_pos);
        }

        return observed_positions;
    }

    // 3. Ä£Äâ´ø¹ý³ÌÔëÉùºÍ²âÁ¿ÔëÉùµÄÔË¶¯
    std::vector<std::pair<Point, Point>> simulateNoisyMotion(double total_time) {
        std::vector<std::pair<Point, Point>> results; // <ÕæÊµÎ»ÖÃ, ¹Û²âÎ»ÖÃ>

        int steps = static_cast<int>(total_time / delta_t);

        Point current_true_pos = initial_position;
        Point current_vel = initial_velocity;

        for (int i = 0; i <= steps; ++i) {
            // Éú³É¹Û²âÎ»ÖÃ£¨´ø²âÁ¿ÔëÉù£©
            Point observed_pos;
            observed_pos.x = current_true_pos.x + measurement_noise(generator);
            observed_pos.y = current_true_pos.y + measurement_noise(generator);

            results.push_back({ current_true_pos, observed_pos });

            // ¸üÐÂËÙ¶È£¨Ìí¼Ó¹ý³ÌÔëÉù£©
            current_vel.x += process_noise(generator);
            current_vel.y += process_noise(generator);

            // ¸üÐÂÎ»ÖÃ
            current_true_pos.x += current_vel.x * delta_t;
            current_true_pos.y += current_vel.y * delta_t;
        }

        return results;
    }

    // Ê¹ÓÃOpenCV¿ÉÊÓ»¯½á¹û
    void visualizeResults(const std::vector<std::pair<Point, Point>>& results,
        const std::string& window_name = "Armor Plate Motion") {
        const int width = 800;
        const int height = 600;
        const double scale = 50.0; // Ëõ·ÅÒò×Ó£¬±ãÓÚÏÔÊ¾

        cv::Mat image(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

        // »æÖÆ×ø±êÖá
        size_t i = 1;
        for (; i < 10; i++){
            cv::line(image, cv::Point(0, i * height / 10), cv::Point(width, i * height / 10), cv::Scalar(0, 0, 0), 1);
            cv::line(image, cv::Point(i * width / 10, 0), cv::Point(i * width / 10, height), cv::Scalar(0, 0, 0), 1);
        }
        //»æÖÆÔ­µã
        putText(image, "O", cv::Point(width / 2 + 5, height / 2 + 15), 3, 0.5, cv::Scalar(0, 0, 0), 2);
        putText(image, "|| 1 X-Div = 1.6 || 1 Y-Div = 1.2 ||",
            cv::Point(width - 310, 20), 3, 0.5, cv::Scalar(0, 0, 0), 2);

        // »æÖÆ¹ì¼£
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& true_pos = results[i].first;
            const auto& observed_pos = results[i].second;

            // ×ª»»×ø±ê£¨Í¼Ïñ×ø±êÏµÓëÊýÑ§×ø±êÏµyÖá·½ÏòÏà·´£©
            cv::Point true_point(width / 2 + true_pos.x * scale, height / 2 - true_pos.y * scale);
            cv::Point observed_point(width / 2 + observed_pos.x * scale, height / 2 - observed_pos.y * scale);

            // »æÖÆÕæÊµÎ»ÖÃ£¨À¶É«£©
            if (i > 0) {
                const auto& prev_true = results[i - 1].first;
                cv::Point prev_true_point(width / 2 + prev_true.x * scale, height / 2 - prev_true.y * scale);
                cv::line(image, prev_true_point, true_point, cv::Scalar(255, 0, 0), 2);
            }
            cv::circle(image, true_point, 3, cv::Scalar(255, 0, 0), -1);

            // »æÖÆ¹Û²âÎ»ÖÃ£¨ºìÉ«£©
            cv::circle(image, observed_point, 3, cv::Scalar(0, 0, 255), -1);

            // Á¬½ÓÕæÊµÎ»ÖÃºÍ¹Û²âÎ»ÖÃ£¨»ÆÉ«£©
            cv::line(image, true_point, observed_point, cv::Scalar(0, 255, 255), 1);
        }

        // Ìí¼ÓÍ¼Àý
        cv::putText(image, "True Position (Blue)", cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);
        cv::putText(image, "Observed Position (Red)", cv::Point(10, 60),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
        cv::putText(image, "Measurement - True (Yellow)", cv::Point(10, 90),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);

        cv::imshow(window_name, image);
        cv::waitKey(0);
    }

    // ´òÓ¡½á¹û
    void printResults(const std::vector<std::pair<Point, Point>>& results, double total_time) {

        //½ûÖ¹Êä³ö·ÇÒì³£INFOÈÕÖ¾
        // æ³¨é‡ŠæŽ‰æ—¥å¿—è®¾ç½®

        cout << "Time\tTrue_X\tTrue_Y\tObs_X\tObs_Y" << endl;
        cout << "--------------------------------------" << endl;

        for (size_t i = 0; i < results.size(); ++i) {
            double current_time = i * delta_t;
            const auto& true_pos = results[i].first;
            const auto& observed_pos = results[i].second;

            cout << fixed << setprecision(3);
            cout << current_time << "\t"
                << true_pos.x << "\t" << true_pos.y << "\t"
                << observed_pos.x << "\t" << observed_pos.y << endl;
        }
    }

    // ×ÛºÏÊä³ö£¨Ö±½Ó¶Ëµ½¶ËÄ¬ÈÏÊä³ö£©
    void SimulateAll() {
        double simulation_time = 1.0; // Ä£Äâ1Ãë

        cout << "=== ×°¼×°åÔË¶¯Ä£Äâ ===" << endl;
        cout << "Ä£ÄâÊ±¼ä: " << simulation_time << " Ãë" << endl;
        cout << "Ö¡ÂÊ: 100fps" << endl;
        cout << "²âÁ¿ÔëÉù±ê×¼²î: 0.5" << endl;
        cout << "¹ý³ÌÔëÉù±ê×¼²î: 0.1" << endl << endl;

        // 1. Ä£ÄâÀíÏëÔË¶¯
        cout << "1. ÀíÏëºã¶¨ËÙ¶ÈÔË¶¯:" << endl;
        auto ideal_positions = simulateIdealMotion(simulation_time);
        cout << "×îÖÕÎ»ÖÃ: (" << ideal_positions.back().x << ", "
            << ideal_positions.back().y << ")" << endl << endl;

        // 2. Ìí¼Ó²âÁ¿ÔëÉù
        cout << "2. ´ø²âÁ¿ÔëÉùµÄ¹Û²â:" << endl;
        auto noisy_observations = addMeasurementNoise(ideal_positions);
        cout << "×îÖÕ¹Û²âÎ»ÖÃ: (" << noisy_observations.back().x << ", "
            << noisy_observations.back().y << ")" << endl;
        cout << "Æ«ÀëÁ¿£º£¨" << abs(noisy_observations.back().x - ideal_positions.back().x)
            << ", " << abs(noisy_observations.back().y - ideal_positions.back().y)
            << ")" << endl;
        cout << "Æ«ÀëÂÊ£º£¨" << 100 * abs(noisy_observations.back().x - ideal_positions.back().x) / ideal_positions.back().x
            << "%, " << 100 * abs(noisy_observations.back().y - ideal_positions.back().y) / ideal_positions.back().y
            << "%)" << endl << endl;

        // 3. Ä£Äâ´ø¹ý³ÌÔëÉùºÍ²âÁ¿ÔëÉùµÄÍêÕûÏµÍ³
        cout << "3. ´ø¹ý³ÌÔëÉùºÍ²âÁ¿ÔëÉùµÄÍêÕûÄ£Äâ:" << endl;
        auto complete_results = simulateNoisyMotion(simulation_time);

        // ´òÓ¡Ç°¼¸¸öÊ±¼ä²½µÄ½á¹û
        printResults(complete_results, simulation_time);

        // ¿ÉÊÓ»¯½á¹û
        visualizeResults(complete_results);
    }
};

class KFSimulator {
private:
    // ÔË¶¯²ÎÊý
    Point initial_position;
    Point initial_velocity;
    double delta_t;

    // ÔëÉù²ÎÊý
    double measurement_noise_std;
    double process_noise_std;

    // Ëæ»úÊýÉú³ÉÆ÷
    std::default_random_engine generator;
    std::normal_distribution<double> measurement_noise;
    std::normal_distribution<double> process_noise;

    // ¿¨¶ûÂüÂË²¨Æ÷
    cv::KalmanFilter kf;

public:
    KFSimulator(double initial_x = 0, double initial_y = 0,
        double velocity_x = 2, double velocity_y = 3,
        double delta_time = 0.01,
        double meas_noise_std = 0.5,
        double proc_noise_std = 0.1)
        : initial_position(initial_x, initial_y)
        , initial_velocity(velocity_x, velocity_y)
        , delta_t(delta_time)
        , measurement_noise_std(meas_noise_std)
        , process_noise_std(proc_noise_std)
        , measurement_noise(0, meas_noise_std)
        , process_noise(0, proc_noise_std)
        , kf(4, 2, 0)  // 4Î¬×´Ì¬Á¿£¬2Î¬¹Û²âÁ¿
    {
        generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
        initKalmanFilter();
    }

private:
    // ³õÊ¼»¯¿¨¶ûÂüÂË²¨Æ÷
    void initKalmanFilter() {
        // ×´Ì¬×ªÒÆ¾ØÕó F (4x4)
        kf.transitionMatrix = (cv::Mat_<float>(4, 4) <<
            1, 0, delta_t, 0,
            0, 1, 0, delta_t,
            0, 0, 1, 0,
            0, 0, 0, 1);

        // ²âÁ¿¾ØÕó H (2x4) - Ö»¹Û²âÎ»ÖÃ
        kf.measurementMatrix = (cv::Mat_<float>(2, 4) <<
            1, 0, 0, 0,
            0, 1, 0, 0);

        // ÕýÈ·¼ÆËã¹ý³ÌÔëÉùÐ­·½²î Q (4x4)
        float velocity_noise_var = process_noise_std * process_noise_std;

        // ¸ù¾ÝÔË¶¯Ä£ÐÍÍÆµ¼Î»ÖÃÔëÉù·½²î
        // Î»ÖÃÔëÉùÓÉËÙ¶ÈÔëÉù»ý·ÖµÃµ½£º¦Ò_pos^2 = ¦Ò_vel^2 * ¦¤t^2
        float position_noise_var = velocity_noise_var * delta_t * delta_t;

        // Î»ÖÃºÍËÙ¶ÈµÄÐ­·½²îÏî£ºcov(pos, vel) = ¦Ò_vel^2 * ¦¤t
        float pos_vel_covar = velocity_noise_var * delta_t;

        kf.processNoiseCov = (cv::Mat_<float>(4, 4) <<
            position_noise_var, 0, pos_vel_covar, 0,
            0, position_noise_var, 0, pos_vel_covar,
            pos_vel_covar, 0, velocity_noise_var, 0,
            0, pos_vel_covar, 0, velocity_noise_var);

        // ²âÁ¿ÔëÉùÐ­·½²î R (2x2)
        float r = measurement_noise_std * measurement_noise_std;
        kf.measurementNoiseCov = (cv::Mat_<float>(2, 2) <<
            r, 0,
            0, r);

        // ºóÑéÎó²îÐ­·½²î P (4x4) - ¾ù³õÊ¼»¯Îª1
        kf.errorCovPost = (cv::Mat_<float>(4, 4) <<
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1);

        // ³õÊ¼×´Ì¬ [x, y, vx, vy]
        kf.statePost = (cv::Mat_<float>(4, 1) <<
            initial_position.x, initial_position.y,
            initial_velocity.x, initial_velocity.y);
    }

public:
    // Ä£Äâ´ø¹ý³ÌÔëÉùºÍ²âÁ¿ÔëÉùµÄÔË¶¯£¬²¢Ó¦ÓÃ¿¨¶ûÂüÂË²¨
    std::vector<std::tuple<Point, Point, Point>> KFsimulate(double total_time) {
        std::vector<std::tuple<Point, Point, Point>> results; // <ÕæÊµÎ»ÖÃ, ¹Û²âÎ»ÖÃ, ÂË²¨Î»ÖÃ>

        int steps = static_cast<int>(total_time / delta_t);

        Point current_true_pos = initial_position;
        Point current_vel = initial_velocity;

        // ³õÊ¼»¯¿¨¶ûÂüÂË²¨Æ÷×´Ì¬
        initKalmanFilter();

        for (int i = 0; i <= steps; ++i) {
            // Éú³ÉÕæÊµÎ»ÖÃ£¨´ø¹ý³ÌÔëÉù£©
            current_vel.x += process_noise(generator);
            current_vel.y += process_noise(generator);
            current_true_pos.x += current_vel.x * delta_t;
            current_true_pos.y += current_vel.y * delta_t;

            // Éú³É¹Û²âÎ»ÖÃ£¨´ø²âÁ¿ÔëÉù£©
            Point observed_pos;
            observed_pos.x = current_true_pos.x + measurement_noise(generator);
            observed_pos.y = current_true_pos.y + measurement_noise(generator);

            // ¿¨¶ûÂüÂË²¨Ô¤²â²½Öè
            kf.predict();

            // ¸üÐÂ²âÁ¿Öµ
            cv::Mat measurement(2, 1, CV_32F);
            measurement.at<float>(0) = observed_pos.x;
            measurement.at<float>(1) = observed_pos.y;

            // ¿¨¶ûÂüÂË²¨Ð£Õý²½Öè
            cv::Mat estimated = kf.correct(measurement);

            // »ñÈ¡ÂË²¨ºóµÄÎ»ÖÃ
            Point filtered_pos(estimated.at<float>(0), estimated.at<float>(1));

            results.push_back({ current_true_pos, observed_pos, filtered_pos });
        }

        return results;
    }

    // ¿ÉÊÓ»¯½á¹û£¨°üº¬¿¨¶ûÂüÂË²¨Ð§¹û£©
    void visualizeKalmanResults(const std::vector<std::tuple<Point, Point, Point>>& results,
        const std::string& window_name = "Kalman Filter Results") {
        const int width = 800;
        const int height = 600;
        const double scale = 50.0;

        cv::Mat image(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

        // »æÖÆ×ø±êÖá
        size_t i = 1;
        for (; i < 10; i++) {
            cv::line(image, cv::Point(0, i * height / 10), cv::Point(width, i * height / 10), cv::Scalar(0, 0, 0), 1);
            cv::line(image, cv::Point(i * width / 10, 0), cv::Point(i * width / 10, height), cv::Scalar(0, 0, 0), 1);
        }
        //»æÖÆÔ­µã
        putText(image, "O", cv::Point(width / 2 + 5, height / 2 + 15), 3, 0.5, cv::Scalar(0, 0, 0), 2);
        putText(image, "|| 1 X-Div = 1.6 || 1 Y-Div = 1.2 ||",
            cv::Point(width - 310, 20), 3, 0.5, cv::Scalar(0, 0, 0), 2);

        // »æÖÆ¹ì¼£
        for (size_t i = 1; i < results.size(); ++i) {
            const auto& prev_true = std::get<0>(results[i - 1]);
            const auto& prev_observed = std::get<1>(results[i - 1]);
            const auto& prev_filtered = std::get<2>(results[i - 1]);

            const auto& true_pos = std::get<0>(results[i]);
            const auto& observed_pos = std::get<1>(results[i]);
            const auto& filtered_pos = std::get<2>(results[i]);

            // ×ª»»×ø±ê
            auto toCVPoint = [&](const Point& p) {
                return cv::Point(width / 2 + p.x * scale, height / 2 - p.y * scale);
                };

            cv::Point prev_true_pt = toCVPoint(prev_true);
            cv::Point true_pt = toCVPoint(true_pos);
            cv::Point observed_pt = toCVPoint(observed_pos);
            cv::Point filtered_pt = toCVPoint(filtered_pos);
            cv::Point prev_filtered_pt = toCVPoint(prev_filtered);

            // »æÖÆÕæÊµ¹ì¼££¨À¶É«£©
            cv::line(image, prev_true_pt, true_pt, cv::Scalar(255, 0, 0), 2);
            cv::circle(image, true_pt, 3, cv::Scalar(255, 0, 0), -1);

            // »æÖÆ¹Û²â¹ì¼££¨ºìÉ«£©
            cv::circle(image, observed_pt, 2, cv::Scalar(0, 0, 255), -1);

            // »æÖÆÂË²¨¹ì¼££¨ÂÌÉ«£©
            cv::line(image, prev_filtered_pt, filtered_pt, cv::Scalar(0, 255, 0), 2);
            cv::circle(image, filtered_pt, 3, cv::Scalar(0, 255, 0), -1);

            // Á¬½Ó¹Û²âºÍÂË²¨Î»ÖÃ£¨»ÆÉ«ÐéÏß£©
            cv::line(image, observed_pt, filtered_pt, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
        }

        // Ìí¼ÓÍ¼Àý
        cv::putText(image, "True Position (Blue)", cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);
        cv::putText(image, "Observed Position (Red)", cv::Point(10, 60),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
        cv::putText(image, "Kalman Filtered (Green)", cv::Point(10, 90),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        cv::putText(image, "Measurement - Filtered (Yellow)", cv::Point(10, 120),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);

        cv::imshow(window_name, image);
        cv::waitKey(0);
    }

    // ´òÓ¡¿¨¶ûÂüÂË²¨½á¹û
    void printKalmanResults(const std::vector<std::tuple<Point, Point, Point>>& results, double total_time) {

        //½ûÖ¹Êä³ö·ÇÒì³£INFOÈÕÖ¾
        // æ³¨é‡ŠæŽ‰æ—¥å¿—è®¾ç½®

        std::cout << "Time\tTrue_X\tTrue_Y\tObserved_X\tObserved_Y\tFiltered_X\tFiltered_Y" << std::endl;
        std::cout << "------------------------------------------------------------------------" << std::endl;

        for (size_t i = 0; i < results.size(); ++i) {
            double current_time = i * delta_t;
            const auto& true_pos = std::get<0>(results[i]);
            const auto& observed_pos = std::get<1>(results[i]);
            const auto& filtered_pos = std::get<2>(results[i]);

            std::cout << std::fixed << std::setprecision(3);
            std::cout << current_time << "\t"
                << true_pos.x << "\t" << true_pos.y << "\t"
                << observed_pos.x << "\t" << observed_pos.y << "\t"
                << filtered_pos.x << "\t" << filtered_pos.y << std::endl;
        }
    }

    // ¼ÆËãÂË²¨ÐÔÄÜÖ¸±ê
    void analyzeFilterPerformance(const std::vector<std::tuple<Point, Point, Point>>& results) {
        double total_obs_error = 0.0;
        double total_filter_error = 0.0;
        int count = results.size();

        for (const auto& result : results) {
            const auto& true_pos = std::get<0>(result);
            const auto& observed_pos = std::get<1>(result);
            const auto& filtered_pos = std::get<2>(result);

            // ¹Û²âÎó²î
            double obs_error = sqrt(pow(observed_pos.x - true_pos.x, 2) +
                pow(observed_pos.y - true_pos.y, 2));
            total_obs_error += obs_error;

            // ÂË²¨Îó²î
            double filter_error = sqrt(pow(filtered_pos.x - true_pos.x, 2) +
                pow(filtered_pos.y - true_pos.y, 2));
            total_filter_error += filter_error;
        }

        cout << "\n=== ¿¨¶ûÂüÂË²¨ÐÔÄÜ·ÖÎö ===" << endl;
        cout << "Æ½¾ù¹Û²âÎó²î: " << total_obs_error / count << endl;
        cout << "Æ½¾ùÂË²¨Îó²î: " << total_filter_error / count << endl;
        cout << "Îó²î¼õÉÙÂÊ: " << (1 - total_filter_error / total_obs_error) * 100 << "%" << endl;
    }

    // ×ÛºÏÊä³ö£¨Ö±½Ó¶Ëµ½¶ËÄ¬ÈÏÊä³ö£©
    void KFSimulateAll() {
        double simulation_time = 1.0; // Ä£Äâ1Ãë
        cout << "=== ´ø¿¨¶ûÂüÂË²¨µÄ×°¼×°åÔË¶¯Ä£Äâ ===" << endl;
        cout << "Ö¡ÂÊ: 100fps" << endl;
        cout << "²âÁ¿ÔëÉù±ê×¼²î: 0.5" << endl;
        cout << "¹ý³ÌÔëÉù±ê×¼²î: 0.1" << endl << endl;
        cout << "Ä£ÄâÊ±¼ä: " << simulation_time << "Ãë" << endl;
        cout << "×´Ì¬Î¬¶È: 4Î¬ [x, y, vx, vy]" << endl;
        cout << "¹Û²âÎ¬¶È: 2Î¬ [x, y]" << endl << endl;

        // ÔËÐÐ´ø¿¨¶ûÂüÂË²¨µÄÄ£Äâ
        auto kalman_results = KFsimulate(simulation_time);

        // ´òÓ¡Ç°¼¸¸öÊ±¼ä²½µÄ½á¹û
        printKalmanResults(kalman_results, simulation_time);

        // ·ÖÎöÐÔÄÜ
        analyzeFilterPerformance(kalman_results);

        // ¿ÉÊÓ»¯½á¹û
        visualizeKalmanResults(kalman_results);
    }
};

int main() {
    // ´´½¨Ä£ÄâÆ÷
    Simulator simulator(0, 0, 2, 3, 0.01, 0.5, 0.1);
    simulator.SimulateAll();
    
    KFSimulator KFsimulator(0, 0, 2, 3, 0.01, 0.5, 0.1);
    KFsimulator.KFSimulateAll();

    return 0;
}