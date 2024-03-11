#pragma once
#ifndef __LOGGER_HPP__
#define __LOGGER_HPP__

#include <iostream>
#include <cstdio>
#include <cstdarg>
#include <string>
#include <ctime>


class Logger
{
    private:
        // Get the current date and time
        static std::string now()
        {
            // Get the current time
            // Put it in following format
            // [YYYY-MM-DD HH:MM:SS]
            time_t rawtime;
            struct tm *timeinfo;
            char buffer[80];
            time(&rawtime);
            timeinfo = localtime(&rawtime);

            strftime(buffer, sizeof(buffer), "[%Y-%M-%d %H:%M:%S]", timeinfo);
            return std::string(buffer);
        }


    public:
        static void log(const char *format, ...)
        {
            std::string str = now() + ": ";
            str += format;
            va_list args;
            va_start(args, format);
            vprintf(str.c_str(), args);
            va_end(args);
        }
        static void warning(const char *format, ...)
        {
            std::string str = now() + ": Warning - ";
            str += format;
            va_list args;
            va_start(args, format);
            vprintf(str.c_str(), args);
            va_end(args);
        }
        static void error(const char *format, ...)
        {
            std::string str = now() + ": Error - ";
            str += format;
            va_list args;
            va_start(args, format);
            vprintf(str.c_str(), args);
            va_end(args);
        }
};




#endif // __LOGGER_HPP__
