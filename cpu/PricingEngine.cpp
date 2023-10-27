#include <cmath>
#include <random>
#include <thread>
#include <climits>
#include <algorithm>
#include <functional>
#include <iostream>


#include "PricingEngine.h"

double generateGaussian(std::normal_distribution<double>& dist, std::mt19937& gen) {
    return dist(gen);
}


inline std::vector<double> getPathStep(double &drift1, double &diffusion1, double &drift2, double &diffusion2,
 const double &corr,std::mt19937& gen1, std::mt19937& gen2, std::normal_distribution<double>& dist) {
    double g1 = generateGaussian(dist, gen1);
    double g2 = generateGaussian(dist, gen2);


    double x1 = g1;
    double x2 = corr * g1 + std::sqrt(1 - corr * corr) * g2;

    // std::cout <<"random variable: "<< x1 <<" "<<x2<<std::endl;
    return {std::exp(drift1 + diffusion1 * x1), std::exp(drift2 + diffusion2 * x2)};
}


inline double discount(double r, double maturity){
    return exp(-r*maturity);
}

int binary_search(const double* lv, double target, int start, int end) {
    int l = start;
    int r = end;

    while (l<=r) {
        int mid = l + (r-l)/2;
        if (lv[mid] == target) {
            return mid;
        } else if (lv[mid] < target) {
            l = mid + 1;
        } else r = mid - 1;
    }
    return std::max(0, r);
}


void generatePath(int threadid, unsigned int seed, double **lowPrice,
    double **finalPrice, const struct Market* market) {

    double drift1;
    double diffusion1;
    int index1;

    double drift2;
    double diffusion2;
    int index2;

    double elapsed = 0.0;

    double maturity = market->maturity;

    double spot1 = market->spots[0];
    double spot2 = market->spots[1];
    double lowp1 = spot1;
    double lowp2 = spot2;
    double strike1 = market->strike[0];
    double strike2 = market->strike[1];

    double r = market->r;

    int num_dts = market->num_dts;
    int num_lv1 = market->num_lv1;
    int num_lv2 = market->num_lv2;

    std::mt19937 mt_rand1(seed + threadid);
    std::mt19937 mt_rand2((seed + threadid + 5)%3);

    std::normal_distribution<double> dist(0.0,1.0);

    // if (threadid == 0)std::cout<<seed + threadid<<std::endl;



    for (int i = 0; i < num_dts; i++) {
        index1 = binary_search(market->lv_index1, (maturity - elapsed)/(log(spot1/strike1) + 0.001), 0, num_lv1-1);
        index2 = binary_search(market->lv_index2, (maturity - elapsed)/(log(spot2/strike2) + 0.001), 0, num_lv2-1);

        // if (threadid==1) std::cout<<index1 << " "<<index2 << " "<<maturity - elapsed<<std::endl;

        drift1 = (r - static_cast<double>(0.5) * market->lv1[i][index1] * market->lv1[i][index1]) * market->dts[i];

        diffusion1 = market->lv1[i][index1] * std::sqrt(market->dts[i]);

        drift2 = (r - static_cast<double>(0.5) * market->lv2[i][index2] * market->lv2[i][index2]) * market->dts[i];

        diffusion2 = market->lv2[i][index2] * std::sqrt(market->dts[i]);

        std::vector<double> exp_value = getPathStep(drift1, diffusion1, drift2, diffusion2, market->corr, mt_rand1, mt_rand2, dist);

        spot1 *= exp_value[0];
        spot2 *= exp_value[1];

        // if (threadid == 0) std::cout<<spot1 << " "<<spot2<<std::endl;


        lowp1 = std::min(lowp1, spot1);
        lowp2 = std::min(lowp2, spot2);

        elapsed += market->dts[i];
    }

    lowPrice[0][threadid] = lowp1;
    lowPrice[1][threadid] = lowp2;

    finalPrice[0][threadid] = spot1;
    finalPrice[1][threadid] = spot2;
}











pricingEngine::pricingEngine(unsigned int seed, unsigned int numSims): m_seed(seed), m_numSims(numSims) {}


pricingEngine::~pricingEngine() {
    for (int i = 0; i < 2; i++){
        delete [] lowPrice[i];
        delete [] finalPrice[i];
    }
}

void pricingEngine::initialize() {
    for (int i = 0; i < 2; i++){
        lowPrice[i] = new double[m_numSims];
        if (lowPrice[i] == nullptr) {
            return;
        }
        finalPrice[i] = new double[m_numSims];
        if (finalPrice[i] == nullptr) {
            return;
        }
        
    }
}



void pricingEngine::simulate(const struct Market* market) {

    std::vector<std::thread> threads;

    for (int i = 0; i < m_numSims; ++i) {
        threads.emplace_back(std::bind(generatePath, i, m_seed, lowPrice, finalPrice, market));
    }

    for (std::thread& t : threads) {
        t.join();
    }
}

void pricingEngine::computeValue(struct Option *option) const {
    double value = 0.0;
    for (int i = 0; i < m_numSims; i++) {
        if (option->barrier1 > lowPrice[0][i] || option->barrier2 > lowPrice[1][i]) continue;
        value += std::min(std::max(finalPrice[0][i] - option->strike1, 0.0), std::max(finalPrice[1][i] - option->strike2, 0.0));
    }

    value /= m_numSims;
    value *= discount(option->r, option->maturity);
    option->value = value;
}

void pricingEngine::computePortValue(struct Portfolio *port) const {
    double value = 0.0;

    int size = port->option_list.size();

    std::vector<std::thread> threads;

    for (int i = 0; i < size; ++i) {
        threads.emplace_back(std::bind(&pricingEngine::computeValue, this, &port->option_list[i]));
    }

    for (std::thread& t : threads) {
        t.join();
    }

    for (int i = 0; i < size; i++) {
        value += port->option_list[i].value;
    }

    port->value = value;
}

