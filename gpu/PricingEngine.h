#ifndef __PRICING_ENGINE_H
#define __PRICING_ENGINE_H

#include "options.h"




class pricingEngine {
    public:
        pricingEngine(unsigned int seed, unsigned int numSims, unsigned int threadBlockSize, unsigned int device);
        void simulate(const struct Market* market);
        void computePortValue(struct Portfolio* port) const;
    private:
        unsigned int m_seed;
        int m_numSims;
        unsigned int m_device;
        unsigned int m_threadBlockSize;
};


#endif // __PRICING_ENGINE_H