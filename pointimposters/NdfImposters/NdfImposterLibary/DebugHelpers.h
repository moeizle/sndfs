#pragma once

#include "glad/glad.h"
#include <string>

void APIENTRY DebugCallback(GLenum source, GLenum type, GLuint id,
    GLenum severity, GLsizei length, const GLchar* message, const void* userParam);


#ifdef _WIN32
std::string getStack();
#endif

#ifdef _LINUX
int getFileAndLine(unw_word_t addr, char *file, size_t flen, int *line);

std::string getStack();
#endif