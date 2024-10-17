/**
 * @file Exceptions.hpp
 * @author Khai Yi Chin (khaiyichin@gmail.com)
 * @brief Exceptions header to provide some commonly used custom exceptions.
 * @version 0.1
 * @date 2024-10-16
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef SVGD_CPP_EXCEPTIONS_HPP
#define SVGD_CPP_EXCEPTIONS_HPP

#include <exception>
#include <string>

#define SVGDCPP_LOG_PREFIX std::string("SVGDCpp: ")

class DimensionMismatchException : public std::exception
{
public:
    explicit DimensionMismatchException(const std::string &message)
        : message_(SVGDCPP_LOG_PREFIX + "[Dimension Error] " + message) {}

    const char *what() const noexcept override
    {
        return message_.c_str();
    }

private:
    std::string message_;
};

class UnsetException : public std::exception
{
public:
    explicit UnsetException(const std::string &message)
        : message_(SVGDCPP_LOG_PREFIX + "[Unset Error] " + message) {}

    const char *what() const noexcept override
    {
        return message_.c_str();
    }

private:
    std::string message_;
};

#endif