#include <iostream>
#include <random>
using std::cout;
using std::endl;

static const int N = 1000;

void print_board(int* queens) {
	int board[N*N] = {};
	for (int q = 0; q < N; ++q)
		board[q * N + queens[q]] = 1;
	cout << endl;
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			if (board[i*N+j])
				cout << "o";
			else
				cout << ".";
		}
		cout << endl;
	}
	cout << endl;
}

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
	int kills = 0;
	const int x = q;
	const int y = queens[q];
	for (int qq = 0; qq < N && kills < N/8; ++qq) {
		if (qq == q)
			continue;
		const int xx = qq;
		const int yy = queens[qq];
		if (yy == y || abs(x-xx) == abs(y-yy)) { // in column or diagonal
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
	int kills = 0;
	const int x = q;
	const int y = queens[q];
	for (int qq = 0; qq < N && kills < N/8; ++qq) {
		if (qq == q)
			continue;
		const int xx = qq;
		const int yy = queens[qq];
		if (yy == y || abs(x-xx) == abs(y-yy)) // in column or diagonal
			kills++;
	}
	return kills;
}

// Genetic approach? quick brain dump
// Use a NN (conv NN ?) to multiply the board or the queens vector by to produce the action.
// add noise to this NN's weights to help to prevent getting stuck in a local minimum?
// or add noise to the output actions instead of weights?

int main() {

	auto eng = std::default_random_engine(12345);
	auto dist = std::uniform_int_distribution<int>(0, N-1);

	int queens[N];
	for (int i = 0; i < N; ++i)
		queens[i] = dist(eng);

	// Kill list allows the code to execute roughly twice as fast :)
	// Which seems to agree with how many loops we have to compute
	// Compute queen kill list
	std::vector<int> kill_list[N];
	for (int i = 0; i < N; ++i)
		queen_kills(i, queens, kill_list);

	// print_board(queens);

	int min_column[N];
	int iterations = 0;
	int kill_sum = 0;
	do {
		// Find positions that give the minimum kill count
		int q = iterations%N;
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

		// Rearrange the kill list
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
		queen_kills(q, queens, kill_list);

		kill_sum = 0;
		for (int i = 0; i < N; ++i)
			kill_sum += kill_list[i].size();
		iterations++;
	} while (kill_sum != 0);

	// loops per game win check
	// previous solution
	//    2N^2
	// current solution
	//    N^2 + N roughly (the maximum is 2N^2 I think, but this would only happen
	//                     if all queens were in the same row)
	// conv solution
	//    N^3 + ... ha, too inefficient and essentially captured by the current impl, but
	//              it's nice to know that a conv net probably could learn the queen_kills
	//              function

	cout << iterations << endl;

	// print_board(queens);

	return 0;
}
