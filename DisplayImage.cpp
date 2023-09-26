#include <opencv2/opencv.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <chrono>
#include <fstream>

// Function to get the pixel value at a specific (x, y) coordinate
void getPixelValue(const cv::Mat image, std::string filename) {
    // Check if the image is valid and if the coordinates are within bounds
    if (image.empty()) {
        std::cerr << "Invalid image or coordinates." << std::endl;
    }
    std::ofstream outputFile(filename);			
    // Access the pixel value at the specified location in the image
    for(int j = 0 ; j < cv::boundingRect(image).height; j++){
    	for(int i = 0 ; i < cv::boundingRect(image).width; i++){
    	    //int pixelValue = (int)image.at<uchar>(y, x);
    	    outputFile << (filename, (int)image.at<uchar>(j, i));
    	    //std::cout << (int)roi.at<uchar>(j,i) << std::endl;
    	}
    }
    // Close the file
    outputFile.close();
}

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

	
    while (true) {
        // Read a frame from the video
        cv::Mat frame;
        int imageCounter = 0; 
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

        // Merge overlapping bounding boxes
        mergedRectangles.clear();
        for (size_t i = 0; i < nonOverlappingRectangles.size(); ++i) {
            cv::Rect mergedRect = nonOverlappingRectangles[i];

            for (size_t j = i + 1; j < nonOverlappingRectangles.size(); ++j) {
                if (areRectanglesOverlapping(mergedRect, nonOverlappingRectangles[j])) {
                    mergedRect |= nonOverlappingRectangles[j]; // Merge rectangles
                }
            }

            mergedRectangles.push_back(mergedRect);
        }
        
	// Save the merged bounding boxes as images
        for (const auto& mergedRect : mergedRectangles) {
            //cv::rectangle(frame, mergedRect, cv::Scalar(0, 255, 0), 2); // Draw green rectangles\
            // Crop the region of interest (ROI) from the frame
            cv::Mat roi = frame(mergedRect);
            
            // Convert the RGB image to grayscale
	    cv::cvtColor(roi, roi, cv::COLOR_BGR2GRAY);
            // Save the cropped ROI as a separate image
            std::string filename_image = "image/frame"+ std::to_string(count_frame) + "object_" + std::to_string(imageCounter) + ".jpg";
            std::string filename_text = "text/frame"+ std::to_string(count_frame) + "object_" + std::to_string(imageCounter) + ".txt";
            getPixelValue(roi, filename_text);
            imageCounter++;
            cv::imwrite(filename_image, roi);
        }
	
        // Display the processed frame with contours and bounding boxes
        cv::imshow("Frame", frame);
        //cv::waitKey(1000);
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
