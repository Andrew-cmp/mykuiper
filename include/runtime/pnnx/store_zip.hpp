#ifndef PNNX_STOREZIP_H
#define PNNX_STOREZIP_H

#include <cstdint>
#include <map>
#include <string>
#include <vector>
namespace pnnx{
class StoreZipReader{
    private:
        FILE *fp;
        struct StoreZipMeta {
            size_t offset;
            size_t size;
        };
        std::map<std::string, StoreZipMeta> filemetas;
    public:
        StoreZipReader();
        ~StoreZipReader();
        int open(const std::string& path);
        size_t get_file_size(const std::string& name);
        int read_file(const std::string& name,char *data);
        int close();
};

class StoreZipWriter{
    public:
        StoreZipWriter();
        ~StoreZipWriter();
        int open(const std::string& path);
        int write_file(const std::string& name, const char* data, size_t size);
        int close();
    
    private:
        FILE *fp;
        struct  StoreZipMeta        {
            std::string name;
            size_t lfh_offset;
            uint32_t crc32;
            uint32_t size;
        };
        std::vector<StoreZipMeta> filemetas;
        

};

}




#endif