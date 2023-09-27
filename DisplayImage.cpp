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

//Function merge intersecting
void mergeIntersectingRectangles(std::vector<cv::Rect>& rectangles);

//Function merge all object
cv::Rect mergeAllObjects(const std::vector<cv::Rect>& objects);

int calNumberofSlidingWindow(int imageWidth, int imageHeight);
 
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
    int count_slidingwindow = 0;
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

#if 0 	//Merge intersecting
        //std::vector<cv::Rect> mergedRectangles;
        //mergeOverlappingRectangles(nonOverlappingRectangles, mergedRectangles);
        
	// Merge intersecting rectangles into one
        mergeIntersectingRectangles(nonOverlappingRectangles);
        
        int imageCounter = 0;
        for (const auto& mergedRect : nonOverlappingRectangles) {
            // Push the processed frame and associated data to the queue
            {
            	// Draw a green rectangle around the detected object
            	cv::rectangle(frame, mergedRect, cv::Scalar(0, 255, 255), 2);
                std::unique_lock<std::mutex> lock(queueMutex);
                frameQueue.push({frame(mergedRect).clone(), count_frame, imageCounter});
                condition.notify_all();
            }
            imageCounter++;
        }
#endif

#if 1 	//Merge all object into 1
        cv::Rect mergedAll = mergeAllObjects(nonOverlappingRectangles);
        std::cout << "Frame_" << count_frame << " width: " << mergedAll.width << " height: " << mergedAll.height << " Sliding window: " << calNumberofSlidingWindow(mergedAll.width, mergedAll.height) << std::endl ;
        count_slidingwindow += calNumberofSlidingWindow(mergedAll.width, mergedAll.height);
        int imageCounter = 0;
    	//cv::rectangle(frame, mergedAll, cv::Scalar(0, 255, 0), 2);
        std::unique_lock<std::mutex> lock(queueMutex);
        frameQueue.push({frame(mergedAll).clone(), count_frame, imageCounter});
        condition.notify_all();
#endif

#if 1
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
    std::cout << "Total sliding window new: " << count_slidingwindow << std::endl;
    std::cout << "Total sliding window old: " << 27960 * count_frame << std::endl;
    std::cout << ((float)count_slidingwindow/(27960.0 * (float)count_frame))*100.0 << std::endl;
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

    // Open the file for writing in binary mode
    std::ofstream outputFile(filename, std::ios::out | std::ios::binary);
    if (!outputFile.is_open()) {
        std::cerr << "Failed to open file for writing." << std::endl;
        return;
    }

    // Write the entire image data to the file at once
    outputFile.write(reinterpret_cast<const char*>(image.data), image.total() * image.elemSize());

    outputFile.close();
}

void mergeIntersectingRectangles(std::vector<cv::Rect>& rectangles) {
    std::vector<cv::Rect> mergedRectangles;

    for (const auto& rect : rectangles) {
        bool merged = false;

        for (auto& mergedRect : mergedRectangles) {
            if (areRectanglesOverlapping(rect, mergedRect)) {
                mergedRect |= rect; // Merge the rectangles
                merged = true;
                break;
            }
        }

        if (!merged) {
            // If the current rectangle didn't merge with any existing rectangles,
            // add it as a new merged rectangle.
            mergedRectangles.push_back(rect);
        }
    }

    // Update the input vector with the merged rectangles.
    rectangles = mergedRectangles;
}

cv::Rect mergeAllObjects(const std::vector<cv::Rect>& objects) {
    if (objects.empty()) {
        // Return an empty rectangle if there are no objects to merge.
        return cv::Rect();
    }

    // Initialize the merged rectangle with the first object.
    cv::Rect mergedRect = objects[0];

    // Iterate through the remaining objects and expand the merged rectangle to cover all objects.
    for (size_t i = 1; i < objects.size(); ++i) {
        mergedRect |= objects[i];
    }

    return mergedRect;
}

int calNumberofSlidingWindow(int imageWidth, int imageHeight){
    int windowWidth = 64;
    int windowHeight = 128;
    int step = 8;
    // Calculate the number of sliding windows in the horizontal and vertical directions
    int horizontalWindows = (imageWidth - windowWidth) / step + 1;
    int verticalWindows = (imageHeight - windowHeight) / step + 1;
    return horizontalWindows * verticalWindows;
}

