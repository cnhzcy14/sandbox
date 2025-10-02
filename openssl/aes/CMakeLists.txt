# add_executable(encryption_openssl main_openssl.cpp)
# target_link_libraries(encryption_openssl PRIVATE OpenSSL::Crypto)

add_executable(decryption_openssl main_decrypt.cpp)
target_link_libraries(decryption_openssl PRIVATE OpenSSL::Crypto)
