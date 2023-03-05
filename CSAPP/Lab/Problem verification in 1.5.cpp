#include <iostream>
#include <ctime>

using namespace std;

#define N 10000 // size of the matrix
int a[N][N]; // Declares a 2D array of size NxN

int main() {

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i][j] = 1; // Fills the array with random numbers from 0 to 9
        }
    }

    clock_t start1 = clock(); // Starts the timer for row-major access
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int res = a[i][j]; // Accesses the array in row-major order
        }
    }
    clock_t end1 = clock(); // Stops the timer for row-major access

    clock_t start2 = clock(); // Starts the timer for column-major access
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            int res = a[i][j]; // Accesses the array in column-major order
        }
    }
    clock_t end2 = clock(); // Stops the timer for column-major access

    cout << "Time to access a[i][j]: " << (double)(end1 - start1) / CLOCKS_PER_SEC << endl;
    cout << "Time to access a[j][i]: " << (double)(end2 - start2) / CLOCKS_PER_SEC << endl;

    system("pause");

    return 0;
}
