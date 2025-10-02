#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <ctime>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <openssl/aes.h>
#include <openssl/evp.h>
#include <openssl/err.h>
#include <openssl/ssl.h>
#include <openssl/crypto.h>

using namespace std;

#define MODEL_HEADER_MAGIC 0x20220312


const uint8_t encrypt_key_aes[16] = {
    0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
    0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c};

typedef struct ABSOLUTELY_TIME_T
{
    uint16_t year;
    uint16_t month;
    uint16_t day;
    uint16_t hour;
    uint16_t minute;
    uint16_t second;
    uint16_t millisecond;
    uint16_t week;
} ABSOLUTELY_TIME;

typedef struct MODEL_HEADER_T
{
    uint32_t magic;
    uint32_t version;
    ABSOLUTELY_TIME abs_time;
    uint32_t len;
    uint8_t model_path[128];
    uint8_t model_name[128];
    uint8_t module_name[128];
    bool isEncryption;
    uint8_t res[130];
    uint8_t sum;
} MODEL_HEADER;



void printModelHead(const MODEL_HEADER &head)
{
    cout << "  magic: 0x" << hex << head.magic << dec << endl;
    cout << "  version: " << head.version << endl;
    cout << "  abs_time:" << endl;
    cout << "    year: " << head.abs_time.year << endl;
    cout << "    month: " << head.abs_time.month << endl;
    cout << "    day: " << head.abs_time.day << endl;
    cout << "    hour: " << head.abs_time.hour << endl;
    cout << "    minute: " << head.abs_time.minute << endl;
    cout << "    second: " << head.abs_time.second << endl;
    cout << "    millisecond: " << head.abs_time.millisecond << endl;
    cout << "    week: " << head.abs_time.week << endl;
    cout << "  len: " << head.len << endl;
    cout << "  model_path: " << reinterpret_cast<const char *>(head.model_path) << endl;
    cout << "  model_name: " << reinterpret_cast<const char *>(head.model_name) << endl;
    cout << "  module_name: " << reinterpret_cast<const char *>(head.module_name) << endl;
    cout << "  isEncryption: " << (head.isEncryption ? "true" : "false") << endl;
    cout << "  ======================================================= " << endl;
}




bool aes_decrypt(const unsigned char *ciphertext, int ciphertext_len, unsigned char *plaintext)
{
    ERR_load_crypto_strings();
    OpenSSL_add_all_algorithms();

    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if (!ctx)
    {
        return false;
    }

    if (1 != EVP_DecryptInit_ex(ctx, EVP_aes_128_ecb(), NULL, encrypt_key_aes, NULL))
    {
        EVP_CIPHER_CTX_free(ctx);
        return false;
    }
    EVP_CIPHER_CTX_set_padding(ctx, 0);

    int len;
    if (1 != EVP_DecryptUpdate(ctx, plaintext, &len, ciphertext, ciphertext_len))
    {
        EVP_CIPHER_CTX_free(ctx);
        return false;
    }

    EVP_CIPHER_CTX_free(ctx);
    EVP_cleanup();
    ERR_free_strings();
    return true;
}


void getFiles(const string &path, vector<string> &files)
{
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(path.c_str())) != NULL)
    {
        while ((ent = readdir(dir)) != NULL)
        {
            if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0)
            {
                continue;
            }

            string full_path = path + "/" + ent->d_name;
            struct stat path_stat;
            stat(full_path.c_str(), &path_stat);

            if (!S_ISDIR(path_stat.st_mode))
            {
                files.push_back(full_path);
            }
        }
        closedir(dir);
    }
}


bool writeDecryptedModelFile(const string &path, const unsigned char *data, size_t data_len)
{
    FILE *pf = fopen(path.c_str(), "wb");
    if (!pf)
    {
        return false;
    }
    bool success = (fwrite(data, 1, data_len, pf) == data_len);
    fclose(pf);
    return success;
}



bool parseModelHeaders(const vector<string>& files, const string &output_path) {
    for (const auto& file : files) {
        FILE* fp = fopen(file.c_str(), "rb");
        if (!fp) {
            printf("Failed to open file: %s\n", file.c_str());
            continue;
        }
        
        MODEL_HEADER header;
        size_t read = fread(&header, sizeof(MODEL_HEADER), 1, fp);
        if (read != 1) {
            printf("Failed to read header from %s\n", file.c_str());
            fclose(fp);
            continue;
        }

        printModelHead(header);
        
        if (header.magic != MODEL_HEADER_MAGIC) {
            printf("Invalid model header magic in %s\n", file.c_str());
            fclose(fp);
            continue;
        }
        
        size_t fsizePad = ((header.len + 15) & ~15); // 对齐到16字节
        vector<unsigned char> fbuff(fsizePad, 0);

        if (fread(fbuff.data(), 1, header.len, fp) != header.len) {
            printf("Failed to read model data from %s\n", file.c_str());
            fclose(fp);
            continue;
        }

        // 解密数据
        vector<unsigned char> decrypted_data(fsizePad);
        if (!aes_decrypt(fbuff.data(), fsizePad, decrypted_data.data()))
        {
            printf("Decryption failed for: %s\n", header.model_name);
            continue;
        }

        string out_path = output_path + "/" + string((char *)header.model_name) + ".bin";
        if (!writeDecryptedModelFile(out_path, decrypted_data.data(), header.len)) {
            printf("Failed to write decrypted model to %s\n", out_path.c_str());
            fclose(fp);
            continue;
        }

        fclose(fp);
    }
    return true;
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        printf("Usage: %s <encrypted_model_directory> [output_directory]\n", argv[0]);
        printf("Decrypts all encrypted model files in the specified directory\n");
        printf("If output_directory is not specified, decrypted files will be saved in the input directory\n");
        return -1;
    }

    string input_path = argv[1];
    string output_path = (argc > 2) ? argv[2] : input_path;

    if (access(input_path.c_str(), F_OK) == -1)
    {
        printf("Input directory does not exist: %s\n", input_path.c_str());
        return -1;
    }

    // 如果指定了输出目录且不存在，则创建它
    if (output_path != input_path && access(output_path.c_str(), F_OK) == -1)
    {
        if (mkdir(output_path.c_str(), 0755) == -1)
        {
            printf("Failed to create output directory: %s\n", output_path.c_str());
            return -1;
        }
    }

    printf("OpenSSL version: %s\n", SSLeay_version(SSLEAY_VERSION));
    printf("Model decryption started. Output will be saved in: %s\n", output_path.c_str());

    vector<string> files;
    vector<string> ownname;

    getFiles(input_path, files);
    parseModelHeaders(files, output_path);

    printf("Model decryption completed\n");
    return 0;
}