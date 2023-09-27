#include <iostream>

int main() {
    // Define the dimensions of the image and the sliding window size
    int imageWidth = 1920;
    int imageHeight = 1080;
    int windowWidth = 64;
    int windowHeight = 128;
    int step = 8;

    // Calculate the number of sliding windows in the horizontal and vertical directions
    int horizontalWindows = (imageWidth - windowWidth) / step + 1;
    int verticalWindows = (imageHeight - windowHeight) / step + 1;

    // Calculate the total number of sliding windows
    int totalWindows = horizontalWindows * verticalWindows;

    std::cout << "Number of sliding windows: " << totalWindows << std::endl;

    return 0;
}

