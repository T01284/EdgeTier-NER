#include "edge_char_encode.h"

#include "edge_char_table.h"

#include <ctype.h>
#include <string.h>

static unsigned char edge_char_id(char ch) {
    const unsigned char u = (unsigned char)ch;
    if (u < EDGE_CHAR_TABLE_SIZE) {
        return kEdgeCharTable[u];
    }
    return EDGE_CHAR_UNK_ID;
}

size_t edge_encode_sentence(
    const char *text,
    float *input_ids,
    uint8_t *mask,
    size_t max_len,
    size_t max_word_len) {
    const size_t total = max_len * max_word_len;
    for (size_t i = 0; i < total; ++i) {
        input_ids[i] = (float)EDGE_CHAR_PAD_ID;
    }
    memset(mask, 0, max_len);

    char word[64];
    size_t word_len = 0;
    size_t token_idx = 0;
    const char *p = text;

    while (*p && token_idx < max_len) {
        while (*p == ' ') {
            ++p;
        }
        if (!*p) {
            break;
        }

        word_len = 0;
        while (*p && *p != ' ' && word_len + 1 < sizeof(word)) {
            word[word_len++] = (char)tolower((unsigned char)*p);
            ++p;
        }
        word[word_len] = '\0';

        mask[token_idx] = 1;
        for (size_t j = 0; j < max_word_len; ++j) {
            const unsigned char cid =
                (j < word_len) ? edge_char_id(word[j]) : EDGE_CHAR_PAD_ID;
            input_ids[token_idx * max_word_len + j] = (float)cid;
        }
        ++token_idx;
    }

    return token_idx;
}
