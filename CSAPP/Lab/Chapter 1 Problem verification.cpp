
//for loop vs while loop in C++


// #include <iostream>
// #include <time.h>

// using namespace std;

// int main()
// {
//     //For loop
//     clock_t tStartFor = clock();
//     for (long long int i=0; i<1000000000; i++) {}
//     double timeFor = (double)(clock() - tStartFor)/CLOCKS_PER_SEC;

//     //While loop
//     clock_t tStartWhile = clock();
//     long long int i = 0;
//     while(i < 1000000000) { i++; }
//     double timeWhile = (double)(clock() - tStartWhile)/CLOCKS_PER_SEC;

//     cout << "Time taken for for loop: " << timeFor << " seconds" << endl;
//     cout << "Time taken for while loop: " << timeWhile << " seconds" << endl;

//     system("pause");
//     return 0;
// }



//switch vs if else ladder in C++

// #include <iostream> 
// #include <ctime>

// using namespace std;
// const int n = 1000000000;
  
// // A utility function to measure time 
// double getTimeTaken(clock_t start, clock_t end) //function for timing 
// { 
//     return (double)(end - start); 
// } 
  
// // Function to compare  the time taken  by switch  & if else 
// void usingSwitch() 
// { 

//     int *arr=new int[n]; 
//     // starting the clock 
//     clock_t start = clock(); 
  
//     // executing loop n times 
//     for (int i = 0; i <= n; i++) { 
//         switch (i%21) { 
//         case 0: 
//             arr[i] = 1; 
//             break; 
//         case 1: 
//             arr[i] = 2; 
//             break; 
//         case 2: 
//             arr[i] = 3; 
//             break; 
//         case 3: 
//             arr[i] = 4; 
//             break;
//         case 4: 
//             arr[i] = 5; 
//             break;
//         case 5:
//             arr[i] = 6;
//             break;
//         case 6:
//             arr[i] = 7;
//             break;
//         case 7:
//             arr[i] = 8;
//             break;
//         case 8:
//             arr[i] = 9;
//             break;
//         case 9:
//             arr[i] = 10;
//             break;
//         case 10:
//             arr[i] = 11;
//             break;
//         case 11:
//             arr[i] = 12;
//             break;
//         case 12:
//             arr[i] = 13;
//             break;
//         case 13:
//             arr[i] = 14;
//             break;
//         case 14:  
//             arr[i] = 15;
//             break;
//         case 15:    
//             arr[i] = 16;
//             break;  
//         case 16:    
//             arr[i] = 17;
//             break;
//         case 17:    
//             arr[i] = 18;
//             break;
//         case 18:    
//             arr[i] = 19;
//             break;  
//         case 19:    
//             arr[i] = 20;
//             break;
//         case 20:    
//             arr[i] = 21;
//             break;
//         } 
//     } 
  
//     // stopping the clock 
//     clock_t end = clock(); 
  
//     cout << "Time taken with switch statement " << getTimeTaken(start, end)<<"\n";
//     delete[] arr;  
// } 
  
// // Function to compare  the time taken by if-else ladder 
// void usingIfElse() 
// { 
  
//     int *arr=new int[n]; 
//     // Starting the clock	 
//     clock_t start1 = clock(); 
//     int i;
//     // Executing loop n times	 
//     for (int k = 0; k < n; k++) {
//         i=k%21;
//         if (i == 0) 
//             arr[i] = 1; 
//         else if (i == 1) 
//             arr[i] = 2; 
//         else if (i == 2) 
//             arr[i] = 3; 
//         else if (i == 3) 
//             arr[i] = 4; 
//         else if (i == 4) 
//             arr[i] = 5; 
//         else if (i == 5) 
//             arr[i] = 6; 
//         else if (i == 6) 
//             arr[i] = 7; 
//         else if (i == 7) 
//             arr[i] = 8; 
//         else if (i == 8) 
//             arr[i] = 9; 
//         else if (i == 9) 
//             arr[i] = 10; 
//         else if (i == 10) 
//             arr[i] = 11; 
//         else if (i == 11) 
//             arr[i] = 12; 
//         else if (i == 12) 
//             arr[i] = 13; 
//         else if (i == 13) 
//             arr[i] = 14; 
//         else if (i == 14) 
//             arr[i] = 15; 
//         else if (i == 15) 
//             arr[i] = 16; 
//         else if (i == 16) 
//             arr[i] = 17; 
//         else if (i == 17) 
//             arr[i] = 18; 
//         else if (i == 18) 
//             arr[i] = 19; 
//         else if (i == 19) 
//             arr[i] = 20; 
//         else if (i == 20) 
//             arr[i] = 21;
//     } 
  
//     // Stopping the clock	 
//     clock_t end1 = clock(); 
  
//     cout << "Time taken with if else ladder " << getTimeTaken(start1, end1);
//     delete[] arr;
// } 
  
// // Driver Function 
// int main() 
// { 
  
//     usingSwitch(); 
//     usingIfElse(); 

//     system("pause");
//     return 0; 
// } 





//pointer vs array in C++


// #include <iostream>
// #include <ctime>
// using namespace std;

// int main()
// {
//     long long int test_count = 1000000000;
//     int *array=new int[test_count];
//     clock_t start, stop;
 
//     // Time using array subscripts
//     start = clock();
//     for (long long int i = 0; i < test_count; ++i)
//         array[i] = 2;
//     stop = clock();
//     double delta_subscript = (stop - start) / (double)CLOCKS_PER_SEC;
 
//     // Time using pointer indexes
//     int* p = array;
//     start = clock();
//     for (long long int i = 0; i < test_count; ++i)
//     {
//         *p = 2;
//         ++p;
//     }
//     stop = clock();
//     double delta_index = (stop - start) / (double)CLOCKS_PER_SEC;
 
//     printf("Array Subscripts: %f \n", delta_subscript);
//     printf("Pointer Indexes: %f \n", delta_index);
    

//     system("pause");
//     return 0;
// }

