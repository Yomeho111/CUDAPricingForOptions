#include <iostream>
#include <vector>
#include <time.h>
#include <stdio.h>
#include <iomanip>

#include "Readfile.h"
#include "PricingEngine.h"
#include "options.h"



int main(){
    struct Portfolio port;
    struct Market market; 
    market.r = 0.03;
    market.corr = 0.5;
    market.is_dts = false;
    market.strike[0] = 100;
    market.strike[1] = 100;
    market.spots[0] = 100;
    market.spots[1] = 100;

    int numSims = 20000;
    int seed = 100;

    Reader reader("../port.csv", "../equity1.csv", "../equity2.csv");

    reader.readMarket(&market);
    reader.readPortfolio(&port, market.maturity, market.r);

    pricingEngine pe(seed, numSims);

    clock_t start_time = clock();

    pe.initialize();

    pe.simulate(&market);
    pe.computePortValue(&port);

    clock_t end_time = clock();

    double milliseconds = (double)(end_time - start_time) / CLOCKS_PER_SEC * 1000.0;


    // for (int i = 0; i < int(port.option_list.size()); i++) {
    //     std::cout<< port.option_list[i].value << std::endl;
    // }

    std::stringstream output;
    output << "Precision:      "
            << "double"<< std::endl;
    output << "Number of sims: " << numSims << std::endl;
    output << std::endl;
    output << "   Spot1   |   Strike1  |   Spot2    |   Strike2  |     r      |   maturity |"
            << std::endl;
    output << "-----------|------------|------------|------------|------------|------------|"
            << std::endl;
    output << std::setw(10) << market.spots[0] << " | ";
    output << std::setw(10) << market.strike[0] << " | ";
    output << std::setw(10) << market.spots[1] << " | ";
    output << std::setw(10) << market.strike[1] << " | ";
    output << std::setw(10) << market.r << " | ";
    output << std::setw(10) << market.maturity << " | ";

    printf("%s\n\n", output.str().c_str());


    std::cout<<"                           The breakdown value for each option "<<std::endl;

    std::cout << "     No      |   barrier1   |   barrier2   |   strike1    |   Strike2    |   value      |"
            << std::endl;
    std::cout << "-------------|--------------|--------------|--------------|--------------|--------------|"
            << std::endl;

    for (int i = 0; i < int(port.option_list.size()); i++) {
        std::cout<< std::setw(12) << i << " | ";
        std::cout<< std::setw(12) << port.option_list[i].barrier1 << " | ";
        std::cout<< std::setw(12) << port.option_list[i].barrier2 << " | ";
        std::cout<< std::setw(12) << port.option_list[i].strike1 << " | ";
        std::cout<< std::setw(12) << port.option_list[i].strike2 << " | ";
        std::cout<< std::setw(12) << port.option_list[i].value << " | ";
        std::cout<< std::endl;
    }
    printf("\n\n");

    printf("The running time is %.2f\n", milliseconds);
    std::cout << "The final value of portfolio is " << port.value<<std::endl;

    return 0;
}