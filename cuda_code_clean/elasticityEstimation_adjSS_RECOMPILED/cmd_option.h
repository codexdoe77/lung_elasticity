#ifndef CMD_OPTION_H
#define CMD_OPTION_H

namespace cmdOption
{
    inline char* get(char ** begin, char ** end, const std::string & option)
    {
        char ** itr = std::find(begin, end, option);
        if (itr != end && ++itr != end)
        {
            return *itr;
        }
        return 0;
    }
}

#endif //CMD_OPTION_H