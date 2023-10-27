#ifndef __READFILE_H
#define __READFILE_H

#include "options.h"
#include <string>

class Reader {
    public:
        Reader(const std::string& path1, const std::string& path2, const std::string& path3):
            portfolioPath(path1),
            equity1Path(path2),
            equity2Path(path3)
            {}
        void readPortfolio(struct Portfolio* port, double maturity, double r) const;
        void readMarket(struct Market* market) const;
    private:
        std::string portfolioPath;
        std::string equity1Path;
        std::string equity2Path;
};


#endif  // __READFILE_H