#include <opencv2/opencv.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <chrono>
#include <fstream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

// Define a structure to hold the processed frames and associated data
struct ProcessedFrame {
    cv::Mat frame;
    int count_frame;
    int imageCounter;
};

// Function to check if two rectangles overlap
bool areRectanglesOverlapping(const cv::Rect& rect1, const cv::Rect& rect2);

// Function to apply background subtraction
void applyBackgroundSubtraction(cv::Ptr<cv::BackgroundSubtractorMOG2> bgSubtractor, const cv::Mat& frame, cv::Mat& frameDiff);

// Function to find non-overlapping bounding boxes
void findNonOverlappingRectangles(const std::vector<std::vector<cv::Point>>& contours, std::vector<cv::Rect>& nonOverlappingRectangles);

// Function to merge overlapping bounding boxes
void mergeOverlappingRectangles(const std::vector<cv::Rect>& nonOverlappingRectangles, std::vector<cv::Rect>& mergedRectangles);

// Function to process video frames
void videoProcessingThread(cv::VideoCapture& video_capture, std::queue<ProcessedFrame>& frameQueue, std::mutex& queueMutex, std::condition_variable& condition);

// Function to save pixel values of an image to a text file
void savePixelValues(const cv::Mat& image, const std::string& filename);

int main() {
    cv::VideoCapture video_capture("/home/kiennt90/Demo_HOG_inCPP/test.mp4");

    if (!video_capture.isOpened()) {
        std::cout << "Error: Could not open video file." << std::endl;
        return -1;
    }

    // Initialize data structures for multithreading
    std::queue<ProcessedFrame> frameQueue;
    std::mutex queueMutex;
    std::condition_variable condition;

    // Create a thread for video frame processing
    std::thread processingThread(videoProcessingThread, std::ref(video_capture), std::ref(frameQueue), std::ref(queueMutex), std::ref(condition));

    // Create a thread for saving frames to files
    std::thread savingThread([&frameQueue, &queueMutex, &condition]() {
        int count_frame = 0;
        while (true) {
            ProcessedFrame processedFrame;
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                condition.wait(lock, [&frameQueue] { return !frameQueue.empty(); });
                processedFrame = frameQueue.front();
                frameQueue.pop();
            }

            if (processedFrame.frame.empty())
                break;

            cv::Mat roi = processedFrame.frame;
            cv::cvtColor(roi, roi, cv::COLOR_BGR2GRAY);
            std::string filename_image = "image/frame" + std::to_string(processedFrame.count_frame) + "object_" + std::to_string(processedFrame.imageCounter) + ".jpg";
            std::string filename_text = "text/frame" + std::to_string(processedFrame.count_frame) + "object_" + std::to_string(processedFrame.imageCounter) + ".txt";
            savePixelValues(roi, filename_text);
            cv::imwrite(filename_image, roi);
        }
    });

    processingThread.join();  // Wait for the processing thread to finish

    // Signal the saving thread to finish
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        frameQueue.push(ProcessedFrame());
        condition.notify_all();
    }

    savingThread.join();  // Wait for the saving thread to finish

    video_capture.release();
    cv::destroyAllWindows();

    return 0;
}

void videoProcessingThread(cv::VideoCapture& video_capture, std::queue<ProcessedFrame>& frameQueue, std::mutex& queueMutex, std::condition_variable& condition) {
    int count_frame = 0;
    cv::Mat previousFrame, frameDiff, threshFrame;
    cv::Ptr<cv::BackgroundSubtractorMOG2> bgSubtractor = cv::createBackgroundSubtractorMOG2();

	auto start_time = std::chrono::high_resolution_clock::now();
    while (true) {
        cv::Mat frame;
        bool ret = video_capture.read(frame);

        if (count_frame % 2 == 0) {
            count_frame++;
            continue;
        }
        count_frame++;

        if (!ret) {
            // Signal the saving thread to finish
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                frameQueue.push(ProcessedFrame());
                condition.notify_all();
            }
            break;
        }

        applyBackgroundSubtraction(bgSubtractor, frame, frameDiff);

        cv::threshold(frameDiff, threshFrame, 30, 255, cv::THRESH_BINARY);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(threshFrame, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        std::vector<cv::Rect> nonOverlappingRectangles;
        findNonOverlappingRectangles(contours, nonOverlappingRectangles);

        std::vector<cv::Rect> mergedRectangles;
        mergeOverlappingRectangles(nonOverlappingRectangles, mergedRectangles);

        int imageCounter = 0;
        for (const auto& mergedRect : mergedRectangles) {
            // Push the processed frame and associated data to the queue
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                frameQueue.push({frame(mergedRect).clone(), count_frame, imageCounter});
                condition.notify_all();
            }
            imageCounter++;
        }
#if 0
        cv::imshow("Frame", frame);
        int keyboard = cv::waitKey(30);
        if (keyboard == 'q' || keyboard == 27) {
            break;
        }
 #endif
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_time = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "Total processing time: " << elapsed_time << " seconds" << std::endl;
    std::cout << "Total frames processed: " << count_frame << std::endl;
}

bool areRectanglesOverlapping(const cv::Rect& rect1, const cv::Rect& rect2) {
    int x1 = std::max(rect1.x, rect2.x);
    int y1 = std::max(rect1.y, rect2.y);
    int x2 = std::min(rect1.x + rect1.width, rect2.x + rect2.width);
    int y2 = std::min(rect1.y + rect1.height, rect2.y + rect2.height);

    return x1 < x2 && y1 < y2;
}

void applyBackgroundSubtraction(cv::Ptr<cv::BackgroundSubtractorMOG2> bgSubtractor, const cv::Mat& frame, cv::Mat& frameDiff) {
    bgSubtractor->apply(frame, frameDiff);
}

void findNonOverlappingRectangles(const std::vector<std::vector<cv::Point>>& contours, std::vector<cv::Rect>& nonOverlappingRectangles) {
    for (const auto& contour : contours) {
        if (cv::contourArea(contour) > 100 && cv::boundingRect(contour).width >= 64 && cv::boundingRect(contour).height >= 128) {
            cv::Rect boundingBox = cv::boundingRect(contour);

            bool overlapping = false;
            for (const auto& mergedRect : nonOverlappingRectangles) {
                if (areRectanglesOverlapping(boundingBox, mergedRect)) {
                    overlapping = true;
                    break;
                }
            }

            if (!overlapping) {
                nonOverlappingRectangles.push_back(boundingBox);
            }
        }
    }
}

void mergeOverlappingRectangles(const std::vector<cv::Rect>& nonOverlappingRectangles, std::vector<cv::Rect>& mergedRectangles) {
    for (size_t i = 0; i < nonOverlappingRectangles.size(); ++i) {
        cv::Rect mergedRect = nonOverlappingRectangles[i];

        for (size_t j = i + 1; j < nonOverlappingRectangles.size(); ++j) {
            if (areRectanglesOverlapping(mergedRect, nonOverlappingRectangles[j])) {
                mergedRect |= nonOverlappingRectangles[j];
            }
        }

        mergedRectangles.push_back(mergedRect);
    }
}

void savePixelValues(const cv::Mat& image, const std::string& filename) {
    if (image.empty()) {
        std::cerr << "Invalid image." << std::endl;
        return;
    }

    std::ofstream outputFile(filename);
    for (int j = 0; j < image.rows; j++) {
        for (int i = 0; i < image.cols; i++) {
            outputFile << (int)image.at<uchar>(j, i) << " ";
        }
        outputFile << "\n";
    }
    outputFile.close();
}
