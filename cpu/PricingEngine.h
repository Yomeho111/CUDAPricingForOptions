#ifndef __PRICING_ENGINE_H
#define __PRICING_ENGINE_H

#include "options.h"




class pricingEngine {
    public:
        pricingEngine(unsigned int seed, unsigned int numSims);
        void initialize();
        void simulate(const struct Market* market);
        void computeValue(struct Option* option) const;
        void computePortValue(struct Portfolio* port) const;
        ~pricingEngine();
    private:
        int m_seed;
        int m_numSims;
        double *lowPrice[2];
        double *finalPrice[2];
};


#endif // __PRICING_ENGINE_H