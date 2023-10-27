#include <fstream>
#include <sstream>
#include <cerrno>
#include <cstdio>
#include <utility>
#include <optional>
#include <algorithm>
#include <iostream>



#include "readfile.h"

int rdn(int y, int m, int d) { /* Rata Die day one is 0001-01-01 */
    if (m < 3)
        y--, m += 12;
    return 365 * y + y / 4 - y / 100 + y / 400 + (153 * m - 457) / 5 + d - 306;
}





bool readOption(struct Portfolio* port, int num, std::string& line, double &maturity, double &r) {
    struct Option opt;
    opt.r = r;
    opt.value = 0;
    opt.maturity = maturity;
    std::string buf;

    std::vector<double> val(6, 0.0);

    std::replace(line.begin(), line.end(), ',', ' ');

    std::istringstream ss(line);

    for (int i = 0; i < 6; i++){
        if (!(ss >> buf))
			return false;
        else {
            std::stringstream sstoi(buf);
			sstoi >> val[i];
			if (sstoi.fail() || !sstoi.eof())
			{
				return false;
			}
        }
    }
    opt.barrier1 = val[1];
    opt.barrier2 = val[2];
    opt.strike1 = val[3];
    opt.strike2 = val[4];

    port->option_list[num] = std::move(opt);
    return true;
}


int read_header(struct Market* market, std::string &line) {
    std::string buf;
    int num_lv;
    int num_dts;

    std::replace(line.begin(), line.end(), ',', ' ');

    std::istringstream ss(line);

    for (int i = 0; i < 5; i++) {
        if (!(ss>>buf)){
            return 0;
        }
        if (i == 2) {
            std::stringstream sstoi(buf);
            sstoi >> num_dts;
            if (sstoi.fail() || !sstoi.eof())
			{
				return 0;
			}
        } else if (i == 4) {
            std::stringstream sstoi(buf);
            sstoi >> num_lv;
            if (sstoi.fail() || !sstoi.eof())
			{
				return 0;
			}
        }
    }
    if (!market->is_dts) {
        market->num_dts = num_dts - 1;
        market->is_dts = true;
    }
        
    return num_lv;
}


std::optional<std::vector<double>> double_lv_index(int num_lv, std::string &line) {
    std::string buf;
    std::vector<double> lv_index(num_lv, 0);


    std::replace(line.begin(), line.end(), ',', ' ');

    std::istringstream ss(line);

    for (int i = 0; i <= num_lv; i++) {
        if (!(ss >> buf)){
            return std::nullopt;
        }
        if (i > 0) {
            std::stringstream sstoi(buf);
            sstoi >> lv_index[i - 1];
            if (sstoi.fail() || !sstoi.eof())
			{
				return std::nullopt;
			}
        }
    }

    return lv_index;
}




int transformDate(std::string& date) {
    std::istringstream ss(date);

    std::string buf;
    int year;
    int month;
    int day;

    for (int i = 0; i < 3; i++) {
        if (!(ss>>buf)) {
            return 0;
        }

        std::stringstream sstoi(buf);
        if (i == 0)
            sstoi >> year;
        else if (i == 1)
            sstoi >> month;
        else
            sstoi >> day;
        if (sstoi.fail() || !sstoi.eof())
        {
            return 0;
        }
    }

    return rdn(year, month, day);
}





void Reader::readPortfolio(struct Portfolio* port, double maturity, double r) const{
    std::ifstream fp(portfolioPath);
    if (!fp.is_open())
	{
		perror("error: ");
		return;
	}

    std::string line;
    getline(fp, line); /*skip the first line*/

    int num = 0;
    while (getline(fp, line)) {
        if (!readOption(port, num, line, maturity, r)) {
            perror("error: ");
            return;
        }
        num++;
    }

    port->size = num;

    fp.close();

}



void Reader::readMarket(struct Market* market) const{
    std::ifstream fp1(equity1Path);

    if (!fp1.is_open())
	{
		perror("error: ");
		return;
	}

    std::string line;
    getline(fp1, line);

    int num_lv1 = read_header(market, line);
    if (num_lv1 == 0) {
        perror("error: ");
        return;
    }
    market->num_lv1 = num_lv1;


    getline(fp1, line);

    std::optional<std::vector<double>> lv1_index = double_lv_index(num_lv1, line);
    if (lv1_index.has_value()){
        for (int i = 0; i < num_lv1; i++) {
            market->lv_index1[i] = lv1_index.value()[i];
        }
    }
    else{
        perror("error: ");
        return;
    }

    /*read the dts and lv surface*/

    std::string buf;
    int startTime = 0;
    int lastTime = 0;
    int currentTime = 0;

    for (int i = 0; i <= market->num_dts; i++) {
        getline(fp1, line);

        std::replace(line.begin(), line.end(), ',', ' ');

        std::istringstream ss(line);

        for (int j = 0; j <= num_lv1; j++) {
            if (!(ss >> buf)){
                perror("error: ");
                return;
            }
            if (j == 0) {
                std::replace(buf.begin(), buf.end(), '/', ' ');
                if (lastTime == 0) {
                    lastTime = transformDate(buf);
                    startTime = lastTime;
                    if (lastTime == 0) {
                        perror("error: ");
                        return;
                    }
                } else {
                    // std::cout << buf <<std::endl;
                    currentTime = transformDate(buf);
                    if (currentTime == 0) {
                        perror("error: ");
                        return;
                    }
                    // std::cout<< "Time is "<<(currentTime - lastTime) <<std::endl;
                    market->dts[i - 1] = (currentTime - lastTime) / 365.0;
                    lastTime = currentTime;
                }
            } else {
                if (i == market->num_dts) break;
                std::stringstream sstoi(buf);
                sstoi >> market->lv1[i][j - 1];
                if (sstoi.fail() || !sstoi.eof())
                {
                    perror("error: ");
                    return;
                }

            }
        }
    }

    fp1.close();

    market->maturity = (currentTime - startTime)/365.0;

    std::ifstream fp2(equity2Path);

    if (!fp2.is_open())
	{
		perror("error: ");
		return;
	}

    getline(fp2, line);

    int num_lv2 = read_header(market, line);
    if (num_lv2 == 0) {
        perror("error: ");
        return;
    }
    market->num_lv2 = num_lv2;


    getline(fp2, line);

    std::optional<std::vector<double>> lv2_index = double_lv_index(num_lv2, line);
    if (lv2_index.has_value()){
        for (int i = 0; i < num_lv2; i++) {
            market->lv_index2[i] = lv2_index.value()[i];
        }
    }
    else{
        perror("error: ");
        return;
    }

    /*read the dts and lv surface*/


    for (int i = 0; i <= market->num_dts; i++) {
        getline(fp2, line);

        std::replace(line.begin(), line.end(), ',', ' ');

        std::istringstream ss(line);

        for (int j = 0; j <= num_lv2; j++) {
            if (!(ss >> buf)){
                perror("error: ");
                return;
            }
            if (j > 0) {
                if (i == market->num_dts) break;
                std::stringstream sstoi(buf);
                sstoi >> market->lv2[i][j - 1];
                if (sstoi.fail() || !sstoi.eof())
                {
                    perror("error: ");
                    return;
                }

            }
        }
    }

    fp2.close();


}



