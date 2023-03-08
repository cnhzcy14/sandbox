#include <iostream>
#include <vector>
#include <set>

int main()
{
    printf("%ld\n", __cplusplus);
    std::cout << "===== " << __cplusplus << std::endl;

    // int sz = 200;
    // std::vector<int> v1;
 
    // auto cap = v1.capacity();
    // std::cout << "initial capacity=" << cap << '\n';
 
    // for (int n = 0; n < sz; ++n) {
    //     v1.push_back(n);
    //     if (cap != v1.capacity()) {
    //         cap = v1.capacity();
    //         std::cout << "new capacity=" << cap << '\n';
    //     }
    // }
    
    // v1.clear();
    // std::cout << "final size=" << v1.size() << '\n';
    // std::cout << "final capacity=" << v1.capacity() << '\n';
    
    std::vector<int> a1{1, 2, 3}, a2{4, 5};

    std::cout << a1.capacity() << a2.capacity() << '\n';
    a1.swap(a2);
    a1.clear();
    std::cout << a1.capacity() << a2.capacity() << '\n';
    
    
    return 0;

}
