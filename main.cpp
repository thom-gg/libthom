#include <stdint.h>
#include <stdio.h>
#include <string>
#include <unordered_map>
#include "enums.h"
#include "routines/routines.h"
#include <stdexcept>


static const std::unordered_map<std::string, RunMode> runModeMap = {
    {"test", RunMode::TEST},
    {"bench", RunMode::BENCH}
};



using RoutineHandler = void (*)(RunMode, int, char **);

static const std::unordered_map<std::string, RoutineHandler> routinesMap = {
    {"reduction", reduction_handler},
    {"sdot", sdot_handler}
};

int main(int argc, char * argv[]) {
    /*
        Params:
            ./libthom <test|bench> <ROUTINE> <implem> ...

        Only parse routine and mode (test|bench) here then pass to routine handler
    */
    if (argc < 3) {
        fprintf(stderr, "Not enough parameters, please provide at least ./libthom <test|bench> <ROUTINE>\n");
        exit(1);
    }

    char * mode = argv[1];
    char * routine = argv[2];

    try {
        RunMode runMode = runModeMap.at(mode);
        RoutineHandler handler = routinesMap.at(routine);
        handler(runMode, argc, argv);
    }
    catch (const std::out_of_range&) {
        fprintf(stderr, "Invalid mode or routine: %s %s\nThey must be registered in main.cpp\n", mode, routine);
        return EXIT_FAILURE;
    }
 



    return EXIT_SUCCESS;


}