#include <stdio.h>
int main()
{
    printf("Hello world");
    return 0;
}

// void age_validation()
// {
//     int age = 0;
//     printf("Enter your age: ");
//     scanf("%d", &age);
//     if (age < 18)
//     {
//         printf("You are not eligible to vote");
//     }
//     else
//     {
//         printf("You are eligible to vote");
//     }
// }

// void postive_negative_zero()
// {
//     int num = 0;
//     printf("Enter a number: ");
//     scanf("%d", &num);
//     if (num > 0)
//     {
//         printf("The number is positive");
//     }
//     else if (num < 0)
//     {
//         printf("The number is negative");
//     }
//     else
//     {
//         printf("The number is zero");
//     }
// }

// void odd_even()
// {
//     int num = 0;
//     printf("Enter a number: ");
//     scanf("%d", &num);
//     if (num % 2 == 0)
//     {
//         printf("The number is even");
//     }
//     else
//     {
//         printf("The number is odd");
//     }
// }

// void leap_year()
// {
//     int year = 0;
//     printf("Enter a year: ");
//     scanf("%d", &year);
//     if (year % 4 == 0)
//     {
//         printf("The year is a leap year");
//     }
//     else
//     {
//         printf("The year is not a leap year");
//     }
// }

// void alphabet_validation()
// {
//     char ch = 'a';
//     printf("Enter a character: ");
//     scanf("%c", &ch);
//     if (ch >= 'a' && ch <= 'z' || ch >= 'A' && ch <= 'Z')
//     {
//         printf("The character is an alphabet");
//     }
//     else
//     {
//         printf("The character is not an alphabet");
//     }
// }

// void divisibleby_5_and_17()
// {
//     int num = 0;
//     printf("Enter a number: ");
//     scanf("%d", &num);
//     if (num % 5 == 0 && num % 17 == 0)
//     {
//         printf(
//             "The number is divisble by 5 and 17");
//     }
//     else
//     {
//         printf(
//             "The number is not divisble by 5 and 17");
//     }
// }

// void for_loop()
// {
//     for (int i = 0; i < 10; i++)
//     {
//         printf("%d\n", i);
//     }
// }

// void while_loop()
// {
//     int i = 0;
//     while (i < 10)
//     {
//         printf("%d\n", i);
//         i++;
//     }
// }

// void do_while_loop()
// {
//     int i = 0;
//     do
//     {
//         printf("%d\n", i);
//         i++;
//     } while (i < 10);
// }

// void infiniteloop()
// {
//     int i = 0;
//     while (i >= 0)
//     {
//         printf("%d\n", i);
//         i++;
//     }
// }

void degree_to_fahrenhiet()
{
    float temp = 0;
    int mode = 0;
    printf("Enter 0 to convert degree to fahrenhiet else 1 to convert fahrenhiet to degree: ");
    scanf("%d", &mode);
    if (mode == 0)
    {
        printf("Enter temperature in degree: ");
        scanf("%f", &temp);
        temp = (temp * 9 / 5) + 32;
        printf("Temperature in fahrenhiet is %f", temp);
    }
    else if (mode == 1)
    {
        printf("Enter temperature in fahrenhiet: ");
        scanf("%f", &temp);
        temp = (temp - 32) * 5 / 9;
        printf("Temperature in degree is %f", temp);
    }
    else
    {
        printf("Invalid mode");
    }
}

void sortlist()
{
    int lengthofarray = 0;
    int arr[100];
    printf("Enter the length of the array: ");
    scanf("%d", &lengthofarray);
    for (int i = 0; i < lengthofarray; i++)
    {
        printf("Enter the element %d: ", i + 1);
        scanf("%d", &arr[i]);
    }
    printf("Sorting array...\n");
    for (int i = 0; i < lengthofarray; i++)
    {
        for (int j = i + 1; j < lengthofarray; j++)
        {
            if (arr[i] < arr[j])
            {
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
    }
    printf("Sorted array: ");
    for (int i = 0; i < lengthofarray; i++)
    {
        printf("%d ", arr[i]);
    }
}