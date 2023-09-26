#include <opencv2/opencv.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <chrono>

// Function to check if two rectangles overlap
bool areRectanglesOverlapping(const cv::Rect& rect1, const cv::Rect& rect2) {
    int x1 = std::max(rect1.x, rect2.x);
    int y1 = std::max(rect1.y, rect2.y);
    int x2 = std::min(rect1.x + rect1.width, rect2.x + rect2.width);
    int y2 = std::min(rect1.y + rect1.height, rect2.y + rect2.height);

    return x1 < x2 && y1 < y2;
}

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
    
    
     // Vector to store merged bounding boxes
    std::vector<cv::Rect> mergedRectangles;
    
    // Record the start time
    auto start_time = std::chrono::high_resolution_clock::now();
	
    cv::Mat previousFrame, frameDiff, threshFrame;
    cv::Ptr<cv::BackgroundSubtractorMOG2> bgSubtractor = cv::createBackgroundSubtractorMOG2();

	int imageCounter = 0; 
    while (true) {
        // Read a frame from the video
        cv::Mat frame;
        bool ret = video_capture.read(frame);
        if(count_frame % 2 == 0){
    		count_frame++;
    		continue;
    	}
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
        
        // Counter for naming the saved images
        
        // Vector to store non-overlapping bounding boxes
        std::vector<cv::Rect> nonOverlappingRectangles;
                // Iterate through the detected bounding boxes
        for (const auto& contour : contours) {
            if (cv::contourArea(contour) > 100 && cv::boundingRect(contour).width >= 64 && cv::boundingRect(contour).height >= 128) {
                cv::Rect boundingBox = cv::boundingRect(contour);

                // Check if this bounding box overlaps with any of the merged bounding boxes
                bool overlapping = false;
                for (const auto& mergedRect : mergedRectangles) {
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

        // Merge all non-overlapping bounding boxes into a single large bounding box
        cv::Rect mergedRect;
        for (const auto& rect : nonOverlappingRectangles) {
            mergedRect |= rect; // Merge rectangles
        }

        // Save the merged bounding box as an image
        cv::Mat roi = frame(mergedRect);
        std::string filename = "image/frame" + std::to_string(count_frame) + "_merged.jpg";
        cv::imwrite(filename, roi);

        // Display the processed frame with merged bounding box
        cv::rectangle(frame, mergedRect, cv::Scalar(0, 255, 0), 2); // Draw
	
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


