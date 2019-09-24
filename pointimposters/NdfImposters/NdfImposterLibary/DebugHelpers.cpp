#include "DebugHelpers.h"

#include <sstream>
#ifndef _WIN32
#include <execinfo.h>
#else
#include <windows.h>
#include <DbgHelp.h>
#pragma comment(lib,"Dbghelp")
#endif

void APIENTRY DebugCallback(GLenum source, GLenum type, GLuint id,
    GLenum severity, GLsizei length, const GLchar* message, const void* userParam) {
    const char *sourceText, *typeText, *severityText;
    switch (source) {
        case GL_DEBUG_SOURCE_API:
            sourceText = "API";
            break;
        case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
            sourceText = "Window System";
            break;
        case GL_DEBUG_SOURCE_SHADER_COMPILER:
            sourceText = "Shader Compiler";
            break;
        case GL_DEBUG_SOURCE_THIRD_PARTY:
            sourceText = "Third Party";
            break;
        case GL_DEBUG_SOURCE_APPLICATION:
            sourceText = "Application";
            break;
        case GL_DEBUG_SOURCE_OTHER:
            sourceText = "Other";
            break;
        default:
            sourceText = "Unknown";
            break;
    }
    switch (type) {
        case GL_DEBUG_TYPE_ERROR:
            typeText = "Error";
            break;
        case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
            typeText = "Deprecated Behavior";
            break;
        case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
            typeText = "Undefined Behavior";
            break;
        case GL_DEBUG_TYPE_PORTABILITY:
            typeText = "Portability";
            break;
        case GL_DEBUG_TYPE_PERFORMANCE:
            typeText = "Performance";
            break;
        case GL_DEBUG_TYPE_OTHER:
            typeText = "Other";
            break;
        case GL_DEBUG_TYPE_MARKER:
            typeText = "Marker";
            break;
        default:
            typeText = "Unknown";
            break;
    }
    switch (severity) {
        case GL_DEBUG_SEVERITY_HIGH:
            severityText = "High";
            break;
        case GL_DEBUG_SEVERITY_MEDIUM:
            severityText = "Medium";
            break;
        case GL_DEBUG_SEVERITY_LOW:
            severityText = "Low";
            break;
        case GL_DEBUG_SEVERITY_NOTIFICATION:
            severityText = "Notification";
            break;
        default:
            severityText = "Unknown";
            break;
    }
    const int outputlength = 8192;
    static char outputstring[outputlength];
    std::string stack = getStack();
#ifdef _WIN32
    sprintf_s(outputstring, outputlength, "[%s %s] (%s %u) %s\nstack trace:\n%s\n", sourceText, severityText, typeText, id, message, stack.c_str());
    OutputDebugStringA(outputstring);
#else
    sprintf(outputstring, "[%s %s] (%s %u) %s\nstack trace:\n%s\n", sourceText, severityText, typeText, id, message, stack.c_str());
#endif
    printf("%s", outputstring);
}

// Implementation taken from OpenGL Insights Code
// https://github.com/OpenGLInsights/OpenGLInsightsCode/blob/master/Chapter%2033%20ARB_debug_output%20A%20Helping%20Hand%20for%20Desperate%20Developers/OpenGLInsightsDebug/vsDebugLib.cpp
// probably by Christophe Riccio

#ifdef _WIN32
std::string getStack() {

    unsigned int   i;
    void         * stack[100];
    unsigned short frames;
    SYMBOL_INFO  * symbol;
    HANDLE         process;
    std::stringstream output;

    process = GetCurrentProcess();

    SymSetOptions(SYMOPT_LOAD_LINES);

    SymInitialize(process, NULL, TRUE);

    frames = CaptureStackBackTrace(0, 200, stack, NULL);
    symbol = (SYMBOL_INFO *)calloc(sizeof(SYMBOL_INFO) + 256 * sizeof(char), 1);
    symbol->MaxNameLen = 255;
    symbol->SizeOfStruct = sizeof(SYMBOL_INFO);

    for (i = 0; i < frames; i++) {
        SymFromAddr(process, (DWORD64)(stack[i]), 0, symbol);
        DWORD  dwDisplacement;
        IMAGEHLP_LINE64 line;

        line.SizeOfStruct = sizeof(IMAGEHLP_LINE64);
        if (!strstr(symbol->Name, "getStack") &&
            !strstr(symbol->Name, "DebugCallback") &&
            SymGetLineFromAddr64(process, (DWORD64)(stack[i]), &dwDisplacement, &line)) {

            output << "function: " << symbol->Name <<
                " - line: " << line.LineNumber << "\n";

        }
        if (0 == strcmp(symbol->Name, "main"))
            break;
    }

    free(symbol);
    return output.str();
}
#endif

#ifdef _LINUX
int getFileAndLine(unw_word_t addr, char *file, size_t flen, int *line) {
    static char buf[256];
    char *p;

    // prepare command to be executed
    // our program need to be passed after the -e parameter
    sprintf(buf, "/usr/bin/addr2line -C -e ./%s -f -i %lx", __progname, addr);

    FILE* f = popen(buf, "r");

    if (f == NULL) {
        perror(buf);
        return 0;
    }

    // get function name
    fgets(buf, 256, f);

    // get file and line
    fgets(buf, 256, f);

    if (buf[0] != '?') {
        int l;
        char *p = buf;

        // file name is until ':'
        while (*p != ':') {
            p++;
        }

        *p++ = 0;
        // after file name follows line number
        strcpy(file, buf);
        sscanf(p, "%d", line);
    } else {
        strcpy(file, "unknown");
        *line = -1;
    }

    pclose(f);
}

std::string getStack() {

    unw_cursor_t cursor;
    unw_context_t uc;
    unw_word_t ip, sp, off;
    unw_proc_info_t pi;
    char file[256], name[256];
    int line;
    int status;

    unw_getcontext(&uc);
    unw_init_local(&cursor, &uc);
    while (unw_step(&cursor) > 0) {
        unw_get_reg(&cursor, UNW_REG_IP, &ip);
        unw_get_reg(&cursor, UNW_REG_SP, &sp);

        unw_get_proc_name(&cursor, name, sizeof(name), &off);
        getFileAndLine((long)ip, file, 256, &line);

        if (line >= 0) {
            char *realname;
            realname = abi::__cxa_demangle(name, 0, 0, &status);

            if (realname) {
                printf("%s: %s, %d\n", realname, file, line);
                free(realname);
            } else {
                printf("%s: %s, %d\n", name, file, line);
            }
        }
    }
}
#endif