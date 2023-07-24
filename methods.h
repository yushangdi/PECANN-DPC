#pragma once

#include <iostream>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>  // For boost::to_lower

enum class Method {
	Doubling, BlindProbe
};

// Overload the stream insertion operator for the Method enum class
std::ostream& operator<<(std::ostream& os, const Method& method) {
    switch (method) {
        case Method::Doubling:
            os << "Doubling";
            break;
        case Method::BlindProbe:
            os << "BlindProbe";
            break;
        default:
            os << "Unknown Method";
            break;
    }
    return os;
}

void validate(boost::any& v,
              const std::vector<std::string>& values,
              Method*, int) {
    namespace po = boost::program_options;

    po::validators::check_first_occurrence(v);

    const std::string& s = po::validators::get_single_string(values);
    std::string lower_s = s;
    boost::to_lower(lower_s);

    if (lower_s == "doubling") {
        v = Method::Doubling;
    } else if (lower_s == "blindprobe") {
        v = Method::BlindProbe;
    } else {
        throw po::validation_error(po::validation_error::invalid_option_value);
    }
}
