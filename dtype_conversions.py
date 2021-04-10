import struct


#################################################################################################
#   Helper functions for converting dtypes in Python (mainly floats -> binary and vice versa)   #
#################################################################################################

# From: https://stackoverflow.com/questions/23624212/how-to-convert-a-float-into-hex
# Converts a float type to a string
def float_to_hex(fl):
    length = 8  # Assuming a 32-bit float, then it should be 8 hex digits
    return hex(struct.unpack('<I', struct.pack('<f', fl))[0])[2:].zfill(length)  # The [2:] removes the 0x prefix


# Convert hex string back to float type
def hex_to_float(hx):
    return struct.unpack('!f', bytes.fromhex(hx))[0]


# Convert hex string to binary string
def hex_to_bin(hx):
    length = len(hx) * 4    # So that we can add leading 0's if needed
    return bin(int(hx, 16))[2:].zfill(length)  # [2:] removes 0b prefix


# Convert binary string to hex string
def bin_to_hex(bn):
    length = int(len(bn) / 4)    # Also for leading 0's (case where binary starts with 0000...)
    return hex(int(bn, 2))[2:].zfill(length)  # [2:] removes 0x prefix


# Uses the functions defined above to turn a float value into a binary string
def float_to_bin(fl):
    return hex_to_bin(float_to_hex(fl))


# Uses functions from above to convert a binary string into a float value
def bin_to_float(bn):
    return hex_to_float(bin_to_hex(bn))


# Verifies the functionality of the 6 functions above
def test_float_hex_bin(flt):
    print("****************** test_float_hex_bin ******************")
    print("Original float value: ", flt, "\t\t", type(flt))
    hex_fl = float_to_hex(flt)
    print("Converted to hex: ", hex_fl, "\t\t\t\t\t\t", type(hex_fl))
    bin_fl = hex_to_bin(hex_fl)
    print("Converted to bin: ", bin_fl, "", type(bin_fl))
    hex_fl2 = bin_to_hex(bin_fl)
    print("Converted back to hex: ", hex_fl2, "\t\t\t\t\t", type(hex_fl2))
    fl2 = hex_to_float(hex_fl2)
    print("And finally back in float: ", fl2, "\t", type(fl2))
    fl3 = bin_to_float(float_to_bin(flt))
    print("Verifying they all work together, your float is still: ", fl3)
    print("********************************************************")