#include <vector>
#include <array>
#include <cstdint>
#include <iostream>


void scratchLayout(std::vector<uint32_t> &levels, std::vector<uint32_t> &offsets, uint32_t N)
{
  if (N == 0)
    return;

  levels.clear();
  levels.push_back((N + 31) / 32);
  while (125 < levels.back())
  {
    levels.push_back((levels.back() + 4) / 5);
  }

  offsets.resize(levels.size() + 4);

  uint32_t o = 0;
  offsets[levels.size()] = o; // Apex
  o += 32 * 4;

  for (int i = static_cast<int>(levels.size()) - 1; 0 < i; i--)
  {
    offsets[i] = o; // HP level i
    o += 4 * levels[i];
  }

  // level zero
  offsets[0] = o;
  o += (levels[0] + 3) & ~3;

  offsets[levels.size() + 1] = o; // Large sideband buffer
  o += levels.empty() ? 0 : (levels[0] + 3) & ~3;

  offsets[levels.size() + 2] = o; // Small sideband buffer
  o += levels.size() < 2 ? 0 : (levels[1] + 3) & ~3;

  offsets[levels.size() + 3] = o; // Final size
  return;
}

void compact(uint32_t N)
{
  if (N == 0)
    return;

  std::vector<uint32_t> levels;
  std::vector<uint32_t> offsets;
  scratchLayout(levels, offsets, N);

  auto L = levels.size();

  std::cout << "N: " << N << std::endl;
  std::cout << "Levels: " << std::endl;
  for (int i = 0; i < levels.size(); i++)
  {
    std::cout << "[" << i << "]: " << levels.at(i) << std::endl;
  }
  std::cout << "Offsets: " << std::endl;
  for (int i = 0; i < offsets.size(); i++)
  {
    std::cout << "[" << i << "]: " << offsets.at(i) << std::endl;
  }

  bool sb = false;

  size_t i = 0;
  if (1 < L)
  {
    std::cout << "reduceBase2b<<<" << (levels[1] + 31) / 32 << ", " << 8 * 32 << ">>>" << std::endl;
    std::cout << "uint  hp2: offsets[1]" << std::endl;
    std::cout << "      sb2: offsets[" << L + 1 + (sb ? 1 : 0) << "]" << std::endl;
    std::cout << "       n2: L[1]" << std::endl;
    std::cout << "uint  hp1: offsets[0]" << std::endl;
    std::cout << "       n1: L[0]" << std::endl;
    std::cout << "      sb0: in_d" << std::endl;
    std::cout << "       n0: N" << std::endl;

    i += 2;
    sb = !sb;
  }
  else if (0 < L)
  {
    std::cout << "reduceBase<<<" << (levels[0] + 3) / 4 << ", " << 4 * 32 << ">>>" << std::endl;
    std::cout << "uint  hp: offsets[0]" << std::endl;
    std::cout << "      sb: offsets[" << L + 1 + (sb ? 1 : 0) << "]" << std::endl;
    std::cout << "      n1: L[0]" << std::endl;
    std::cout << "     src: in_d" << std::endl;
    std::cout << "      n0: N" << std::endl;

    i += 1;
    sb = !sb;
  }

  for (; i + 2 < L; i += 3)
  {
    std::cout << "reduce3<<<" << (levels[i + 2] + 31) / 32 << ", " << 160 << ">>>" << std::endl;
    std::cout << "uint4 hp3: offsets[" << i + 2 << "]" << std::endl;
    std::cout << "      sb3: offsets[" << L + 1 + (sb ? 1 : 0) << "]" << std::endl;
    std::cout << "       n3: L[" << i + 2 << "]" << std::endl;
    std::cout << "uint4 hp2: offsets[" << i + 1 << "]" << std::endl;
    std::cout << "       n2: L[" << i + 1 << "]" << std::endl;
    std::cout << "uint4 hp1: offsets[" << i << "]" << std::endl;
    std::cout << "       n1: L[" << i << "]" << std::endl;
    std::cout << "      sb0: offsets[" << L + 1 + (sb ? 0 : 1) << "]" << std::endl;
    std::cout << "       n0: L[" << i - 1 << "]" << std::endl;

    sb = !sb;
  }

  for (; i + 1 < L; i += 2)
  {
    std::cout << "reduce2<<<" << (levels[i + 1] + 31) / 32 << ", " << 160 << ">>>" << std::endl;
    std::cout << "uint4 hp2: offsets[" << i + 1 << "]" << std::endl;
    std::cout << "      sb2: offsets[" << L + 1 + (sb ? 1 : 0) << "]" << std::endl;
    std::cout << "       n2: L[" << i + 1 << "]" << std::endl;
    std::cout << "uint4 hp1: offsets[" << i << "]" << std::endl;
    std::cout << "       n1: L[" << i << "]" << std::endl;
    std::cout << "      sb0: offsets[" << L + 1 + (sb ? 0 : 1) << "]" << std::endl;
    std::cout << "       n0: L[" << i - 1 << "]" << std::endl;

    sb = !sb;
  }

  for (; i < L; i++)
  {
    std::cout << "reduce1<<<" << (levels[i] + 31) / 32 << ", " << 160 << ">>>" << std::endl;
    std::cout << "uint4 hp1: offsets[" << i << "]" << std::endl;
    std::cout << "      sb1: offsets[" << L + 1 + (sb ? 1 : 0) << "]" << std::endl;
    std::cout << "       n1: L[" << i << "]" << std::endl;
    std::cout << "      sb0: offsets[" << L + 1 + (sb ? 0 : 1) << "]" << std::endl;
    std::cout << "       n0: L[" << i - 1 << "]" << std::endl;

    sb = !sb;
  }

  std::cout << "reduceApex<<<" << 1 << ", " << 128 << ">>>" << std::endl;
  std::cout << "uint4 apex: scratch_d" << std::endl;
  std::cout << "       sum: sum_d" << std::endl;
  std::cout << "       sb0: offsets[" << L + 1 + (sb ? 0 : 1) << "]" << std::endl;
  std::cout << "        n0: L[" << L - 1 << "]" << std::endl;
  return;
}

void runCompactTest(uint32_t N, uint32_t m)
{
  uint32_t sum;
  std::vector<uint32_t> out, in;
  // buildCompactProblemBestCase(out, sum, in, N, m);
  compact(N);
}

int main()
{
  for (uint32_t N = 1; N < (uint64_t)(101010 / (sizeof(uint32_t) * 4)); N = 3 * N + N / 3)
  {
    for (uint32_t m = 32; m < 512; m *= 2)
    {
      std::cout << "\n====================" << std::endl;
      runCompactTest(N, m);
    }
  }
}