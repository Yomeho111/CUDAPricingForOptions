#ifndef __OPTIONS_H
#define __OPTIONS_H

#include <vector>

struct Option {
    double barrier1;
    double barrier2;
    double strike1;
    double strike2;

    double maturity;
    double value;
    double r;
};


struct Portfolio {
    struct Option option_list[1000];
    double value = 0.0;
    int size;
};



struct Market {
    double r;
    double spots[2];
    double strike[2];

    int num_dts;
    double dts[10000];

    int num_lv1;
    int num_lv2;
    double lv_index1[10000];
    double lv_index2[10000];
    double lv1[10000][10000];
    double lv2[10000][10000];

    double maturity;

    double corr;

    bool is_dts;

};




#endif // __OPTIONS_H