#include <filesystem>
#include <string>
#include <iostream>

namespace fs = std::filesystem;

int main()
{
    const std::string path = "newF";
    for (int i = 1; i <= 10; ++i)
    {
        try
        {
            if (fs::create_directory(path + std::to_string(i)))
                std::cout << "Created a directory\n";
            else
                std::cerr << "Failed to create a directory\n";
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
        }
    }
    return 0;
}