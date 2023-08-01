#include <iostream>
#include <chrono>
#include <thread>

namespace sc = std::chrono;

int main()
{
    auto start = sc::system_clock::now();
    // Some computation here
    sc::milliseconds timespan(1605); // or whatever
    std::this_thread::sleep_for(timespan);
    auto end = sc::system_clock::now();

    double elapsed_miliseconds =
        1.e-3 * sc::duration_cast<sc::microseconds>(end - start).count();

    // std::chrono::duration<double> elapsed_miliseconds = end - start;
    std::time_t end_time = sc::system_clock::to_time_t(end);

    std::cout << "finished computation at " << std::ctime(&end_time)
              << "elapsed time: " << elapsed_miliseconds << "ms"
              << std::endl;
}
