#include <iostream>
#include <condition_variable>
#include <thread>
#include <chrono>

#define CAP 3

enum StatusType{ EMPTY = 0, STAGING = 1, SAVING = 2};
int event[CAP] = {0};
std::condition_variable cv[CAP];
std::mutex cv_m[CAP]; 


void waits(int i)
{
    while (1)
    {
        std::unique_lock<std::mutex> lk(cv_m[i]);
        std::cerr << "Waiting... \n";
        cv[i].wait(lk, [&] { return event[i] == 1; });
        std::cerr << "...finished waiting. " << i << "== 1\n";
        lk.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        event[i] = 0;
    }
}

void signals()
{
    for (int i = 0; i < 600000; i++)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        {
            std::lock_guard<std::mutex> lk(cv_m[i % CAP]);
            event[i % CAP] = 1;
            std::cerr << "Notifying again..." << i << "\n";
        }
        cv[i % CAP].notify_all();
    }
}

int main()
{
    std::thread t1(waits, 0), t2(waits, 0), t3(waits, 0), t4(signals);
    t1.join();
    t2.join();
    t3.join();
    t4.join();
}