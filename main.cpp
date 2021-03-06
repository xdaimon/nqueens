#include <iostream>
#include <vector>
#include <random>

// g++ -std=c++11 -O3 main.cpp -o main

static const int N = 8;

/*
Could use a zero padded convolution with the kernel:
       ...
     1001001
     0101010
     0011100
 ... 111N111 ...
     0011100
     0101010
     1001001
       ...
followed by a pointwise application of max(0, x - N)
followed by a sum of all elements
so
win if 0 == Sum(Max(Conv(board, kernel) - N * ones_matrix, 0))
*/

int queen_kills(int q, int* queens, std::vector<int>* kill_list) {
	// q,         the index of a queen in the queens array, the queen id, represents the row of the queen on the board
	// queens,    the queens array, the array of queen column positions, length = N
	// kill_list, an array of N lists of queen ids. The nth list holds the queen ids that the nth queen can kill
	//
	// returns the number of queens that queen q can strike

	int kills = 0;
	const int x = q;
	const int y = queens[q];
	for (int qq = 0; qq < N; ++qq) {
		if (qq == q)
			continue;
		const int xx = qq;
		const int yy = queens[qq];
		// Check if qq and q are in the same column or if qq and q are in the same diagonal
		if (yy == y || abs(x-xx) == abs(y-yy)) {
			kills++;
			// add qq to kill list of q
			// and q to kill list of qq
			kill_list[q].push_back(qq);
			kill_list[qq].push_back(q);
		}
	}
	return kills;
}

int queen_kills(int q, int* queens) {
	// q,         the index of a queen in the queens array, the queen id, represents the row of the queen on the board
	// queens,    the queens array, the array of queen column positions, length = N
	//
	// returns the number of queens that queen q can strike

	int kills = 0;
	const int x = q;
	const int y = queens[q];
	for (int qq = 0; qq < N; ++qq) {
		if (qq == q)
			continue;
		const int xx = qq;
		const int yy = queens[qq];
		if (yy == y || abs(x-xx) == abs(y-yy)) // if in column else if in diagonal
			kills++;
	}
	return kills;
}

// Genetic approach? quick brain dump
// Use a NN (conv NN ?) to multiply the board or the queens vector by to produce the action.
// add noise to this NN's weights to help to prevent getting stuck in a local minimum?
// or add noise to the output actions instead of weights?

int main() {

	// Setup random numbers in range [0, N-1]
	auto eng = std::default_random_engine(12345);
	auto dist = std::uniform_int_distribution<int>(0, N-1);

	// Initialize queens
	int queens[N];
	for (int& q : queens)
		q = dist(eng);

	// Compute queen kill list
	// i.e. what queens can the nth queen strike?
	std::vector<int> kill_list[N];
	for (int& q : queens)
		queen_kills(q, queens, kill_list);

	int min_column[N];
	int iterations = 0;
	int kill_sum;
	do {
		// Choose the qth queen
		int q = iterations % N;

		// Find column positions that give the minimum kill count for the qth queen
		// If there are multiple positions that give a minimum, then choose one at random
		int min_kills = std::numeric_limits<int>::max();
		int num_min_columns = 0;
		int kills;
		for (int col = 0; col < N; ++col) {
			kills = 0;
			queens[q] = col;
			kills = queen_kills(q, queens);
			if (kills <= min_kills) {
				if (kills != min_kills)
					num_min_columns = 0;
				min_kills = kills;
				min_column[num_min_columns] = col;
				num_min_columns++;
			}
		}

		// Move queen and make sure the kill lists are correct
		{
			// Remove q from any kill lists
			std::vector<int>& qkl = kill_list[q];
			for (size_t i = 0; i < qkl.size(); ++i) {
				std::vector<int>& qqkl = kill_list[qkl[i]];
				for (size_t j = 0; j < qqkl.size(); ++j) {
					if (qqkl[j] == q) {
						qqkl.erase(qqkl.begin()+j);
						break;
					}
				}
			}
			qkl.clear();
			// Place queen at a position that allows a minimum number of kills
			queens[q] = min_column[dist(eng) % num_min_columns];
			// Add q to kill list of queens who can kill q
			queen_kills(q, queens, kill_list);
		}

		// Compute the end game condition
		{
			kill_sum = 0;
			for (auto& v : kill_list)
				kill_sum += v.size();
		}

		iterations++;
	} while (kill_sum != 0);

	// The kill list allows the code to execute roughly two times faster than previous solution :)
	// loops per iteration
	// previous solution
	//    2N^2
	// current solution
	//    N^2 + N roughly (the maximum is 2N^2 I think, but this would only happen
	//                     if all queens were in the same row?)
	// conv solution
	//    N^3 + ... ha, too inefficient and essentially captured by the current impl, but
	//              it's nice to know that a conv net probably could learn the queen_kills
	//              function

	std::cout << iterations << std::endl;

	return 0;
}
