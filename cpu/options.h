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
    std::vector<struct Option> option_list;
    double value = 0.0;
};



struct Market {
    double r;
    double spots[2];
    double strike[2];

    int num_dts;
    double dts[500];

    int num_lv1;
    int num_lv2;
    double lv_index1[500];
    double lv_index2[500];
    double lv1[500][500];
    double lv2[500][500];

    double maturity;

    double corr;

    bool is_dts;

};




#endif // __OPTIONS_H