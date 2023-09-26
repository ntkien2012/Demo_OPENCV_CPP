#include <opencv2/opencv.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <chrono>

int main() {
    // Open the MP4 file
    //	std::cout << "cv information: " << cv::getBuildInformation() << std::endl;
	cv::VideoCapture video_capture("/home/kiennt90/Demo_HOG_inCPP/test.mp4");

    // Check if the video file was opened successfully
    if (!video_capture.isOpened()) {
        std::cout << "Error: Could not open video file." << std::endl;
        return -1;
    }

    // Get the FPS (frames per second) of the video
    int fps = static_cast<int>(video_capture.get(cv::CAP_PROP_FPS));

    int count_frame = 0;

    // Record the start time
    auto start_time = std::chrono::high_resolution_clock::now();
	
    cv::Mat previousFrame, frameDiff, threshFrame;
    cv::Ptr<cv::BackgroundSubtractorMOG2> bgSubtractor = cv::createBackgroundSubtractorMOG2();

    while (true) {
        // Read a frame from the video
        cv::Mat frame;
        bool ret = video_capture.read(frame);
        count_frame++;
	
        // Check if the frame was read successfully
        if (!ret) {
            break; // End of video
        }
        
        // Apply background subtraction
        bgSubtractor->apply(frame, frameDiff);
        
        // Threshold the difference image
        cv::threshold(frameDiff, threshFrame, 30, 255, cv::THRESH_BINARY);
        
        // Find contours
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(threshFrame, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        // Draw rectangles around moving objects
        for (const auto& contour : contours) {
            if (cv::contourArea(contour) > 100 ) { // Filter out small contours
                cv::Rect boundingBox = cv::boundingRect(contour);
                if(boundingBox.width >= 128 and boundingBox.height >= 64){
                	cv::rectangle(frame, boundingBox, cv::Scalar(0, 255, 0), 2);
               	}
            }
        }
        
        //cv:imwrite("image/output"+ std::to_string(count_frame) + ".jpg", frame);

        // Display the processed frame with contours and bounding boxes
        cv::imshow("Frame", frame);

        // Check for user input to exit the program (press 'q' or ESC)
        int keyboard = cv::waitKey(30);
        if (keyboard == 'q' || keyboard == 27) {
            break;
        }
    }

    // Record the end time
    auto end_time = std::chrono::high_resolution_clock::now();

    // Calculate and print the elapsed time
    double elapsed_time = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "Total processing time: " << elapsed_time << " seconds" << std::endl;

    // Print the total number of frames processed
    std::cout << "Total frames processed: " << count_frame << std::endl;

    // Release the video capture object and close any open windows
    video_capture.release();
    cv::destroyAllWindows();

    return 0;
}

