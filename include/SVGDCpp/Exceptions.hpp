/**
 * @file Exceptions.hpp
 * @author Khai Yi Chin (khaiyichin@gmail.com)
 * @brief Exceptions header to provide some commonly used custom exceptions.
 * 
 * @copyright Copyright (c) 2024 Khai Yi Chin
 * 
 */

#ifndef SVGDCPP_EXCEPTIONS_HPP
#define SVGDCPP_EXCEPTIONS_HPP

#include <exception>
#include <string>

#define SVGDCPP_LOG_PREFIX std::string("SVGDCpp: ") ///< Convenience macro to prefix output logs.

/**
 * @class DimensionMismatchException
 * @brief Exception for dimension mismatch type errors.
 * @ingroup Core_Module
 */
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

/**
 * @class UnsetException
 * @brief Exception for unset value type errors.
 * @ingroup Core_Module
 */
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