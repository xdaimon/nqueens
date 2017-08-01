#include <iostream>
#include <random>
using std::cout;
using std::endl;

static const int N = 20;

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

int queen_kills(int q, int* queens) {
	int kills = 0;
	const int x = q;
	const int y = queens[q];
	for (int qq = 0; qq < N && kills < N/4; ++qq) {
		if (qq == q)
			continue;
		const int xx = qq;
		const int yy = queens[qq];
		if (yy == y) // in column
			kills++;
		else if (fabs((x-xx)/float(y-yy)) == 1.f) // on a diagonal
			kills++;
	}
	return kills;
}

// Genetic approach.
// Use a NN (conv NN ?) to multiply the board or the queens vector by to produce the action.
// add noise to this NN's weights to help to prevent getting stuck in a local minimum?
// or add noise to the output actions instead of weights?

int main() {

	auto eng = std::default_random_engine(12345);
	auto dist = std::uniform_int_distribution<int>(0, N-1);

	int queens[N];
	for (int i = 0; i < N; ++i)
		queens[i] = dist(eng);

	// print_board(queens);

	int min_column[N];
	int iterations = 0;
	int kill_sum = 0;
	do {
		kill_sum = 0;
		for (int q = 0; q < N; ++q) {
			// Find positions that give the minimum kill count
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
			// Place queen at a position that allows a minimum number of kills
			queens[q] = min_column[dist(eng) % num_min_columns];
			kill_sum += min_kills;
			iterations++;
		}
	} while (kill_sum != 0);

	// loops per game win check
	// first solution
	//    about 1000 * 1000 * 1000
	// second solution
	//    nxnMult * n + nSum + nSub + nMax

	cout << iterations << endl;

	print_board(queens);

	return 0;
}
