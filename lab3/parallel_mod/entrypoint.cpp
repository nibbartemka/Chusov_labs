#include "vector_mod.h"
#include "test.h"
#include "performance.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include "num_threads.h"

int main(int argc, char** argv)
{
    // Открываем CSV файл для записи
    std::ofstream csv_file("vector_mod_output.csv");
    if (!csv_file.is_open())
    {
        std::cerr << "Error: Could not open CSV file for writing.\n";
        return -1;
    }
    
    // Записываем заголовок CSV
    csv_file << "Threads,Value_Hex,Duration_ms,Acceleration\n";
    
    std::cout << "==Correctness tests. ";
    for (std::size_t iTest = 0; iTest < test_data_count; ++iTest)
    {
        if (test_data[iTest].result != vector_mod(test_data[iTest].dividend, test_data[iTest].dividend_size, test_data[iTest].divisor))
        {
            std::cout << "FAILURE==\n";
            csv_file.close();
            return -1;
        }
    }
    std::cout << "ok.==\n";
    
    std::cout << "==Performance tests. ";
    auto measurements = run_experiments();
    std::cout << "Done==\n";
    
    // Вывод в консоль
    std::cout << std::setfill(' ') << std::setw(2) << "T:" << " |" << std::setw(3 + 2 * sizeof(IntegerWord)) << "Value:" << " | "
              << std::setw(14) << "Duration, ms:" << " | Acceleration:\n";
    
    // Вывод в файл и консоль
    for (std::size_t T = 1; T <= measurements.size(); ++T)
    {
        // Форматируем данные
        std::ostringstream hex_value;
        hex_value << "0x" << std::setw(2 * sizeof(IntegerWord)) << std::setfill('0') 
                  << std::hex << measurements[T - 1].result;
        
        double acceleration = static_cast<double>(measurements[0].time.count()) / measurements[T - 1].time.count();
        
        // Вывод в консоль
        std::cout << std::setw(2) << T << " | " << hex_value.str();
        std::cout << " | " << std::setfill(' ') << std::setw(14) << std::dec << measurements[T - 1].time.count();
        std::cout << " | " << acceleration << "\n";
        
        // Запись в CSV файл
        csv_file << T << ","
                 << hex_value.str() << ","
                 << measurements[T - 1].time.count() << ","
                 << std::fixed << std::setprecision(6) << acceleration << "\n";
    }
    
    csv_file.close();
    std::cout << "\nResults saved to vector_mod_output.csv\n";
    
    return 0;
}