#include <iostream>
#include <fstream>
#include <string>
#include <vector>

int main()
{
    // The base filename with "%d" representing the index
    const std::string baseFilename = "/home/radxa/work/data/700/pointsData_/points%03d.txt";

    // // Vector to store the generated filenames
    // std::vector<std::string> filenames;

    // // Number of files to generate
    // const int numFiles = 5;

    // for (int i = 0; i < numFiles; ++i) {
    //     char filenameBuffer[100]; // Adjust buffer size as needed
    //     std::sprintf(filenameBuffer, baseFilename.c_str(), i);
    //     filenames.push_back(filenameBuffer);
    // }

    // // Display the generated filenames
    // for (const auto& filename : filenames) {
    //     std::cout << filename << std::endl;
    // }

    std::ofstream ofs; // output file stream
    
    for(size_t i = 0; ;i++)
    {
        char filenameBuffer[1024]; // Adjust buffer size as needed
        std::sprintf(filenameBuffer, baseFilename.c_str(), i);

        std::cout << std::string(filenameBuffer) << std::endl;
        ofs.open(filenameBuffer, std::fstream::in); // set file cursor at the end

        if (ofs)
        {
            ofs.close();
        }
        else
        {
            break;
        }
    }

    return 0;
}