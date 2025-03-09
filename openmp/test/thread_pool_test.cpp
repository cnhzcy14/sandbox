#include <iostream>
#include <functional>
#include <vector>
using namespace std;

void fun(int a, int b)
{
	cout << "fun exec :" << a << '+' << b << '=' << a + b << endl;
}

class C
{
private:
	float m_c = 2.0f;

public:
	void mp(float d)
	{
		cout << "c::mp exec :" << m_c << 'x' << d << '=' << m_c * d << endl;
	}
};

int main(int argc, char *argv[])
{
	const int task_groups = 5;
	C c[task_groups];
	vector<function<void(void)>> tasks;
	for (int i = 0; i < task_groups; ++i)
	{
		tasks.push_back(bind(fun, 10, i * 10));
		tasks.push_back(bind(&C::mp, &c[i], i * 2.0f));
		tasks.push_back(bind(
			[=](void)
			{ cout << "lambada :" << i << endl; }));
	}
	size_t sz = tasks.size();
#pragma omp parallel for
	for (size_t i = 0; i < sz; ++i)
	{
		tasks[i]();
	}
	return 0;
}
