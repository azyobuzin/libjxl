/* Copyright (c) the JPEG XL Project Authors. All rights reserved.
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

/** @addtogroup libjxl_encoder
 * @{
 * @file encode.h
 * @brief Encoding API for JPEG XL.
 */

#ifndef JXL_ENCODE_H_
#define JXL_ENCODE_H_

#include "jxl/decode.h"
#include "jxl/jxl_export.h"
#include "jxl/memory_manager.h"
#include "jxl/parallel_runner.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

/**
 * Encoder library version.
 *
 * @return the encoder library version as an integer:
 * MAJOR_VERSION * 1000000 + MINOR_VERSION * 1000 + PATCH_VERSION. For example,
 * version 1.2.3 would return 1002003.
 */
JXL_EXPORT uint32_t JxlEncoderVersion(void);

/**
 * Opaque structure that holds the JPEG XL encoder.
 *
 * Allocated and initialized with JxlEncoderCreate().
 * Cleaned up and deallocated with JxlEncoderDestroy().
 */
typedef struct JxlEncoderStruct JxlEncoder;

/**
 * Opaque structure that holds frame specific encoding options for a JPEG XL
 * encoder.
 *
 * Allocated and initialized with JxlEncoderOptionsCreate().
 * Cleaned up and deallocated when the encoder is destroyed with
 * JxlEncoderDestroy().
 */
typedef struct JxlEncoderOptionsStruct JxlEncoderOptions;

/**
 * Return value for multiple encoder functions.
 */
typedef enum {
  /** Function call finished successfully, or encoding is finished and there is
   * nothing more to be done.
   */
  JXL_ENC_SUCCESS = 0,

  /** An error occurred, for example out of memory.
   */
  JXL_ENC_ERROR = 1,

  /** The encoder needs more output buffer to continue encoding.
   */
  JXL_ENC_NEED_MORE_OUTPUT = 2,

  /** The encoder doesn't (yet) support this.
   */
  JXL_ENC_NOT_SUPPORTED = 3,

} JxlEncoderStatus;

/**
 * Id of per-frame options to set to JxlEncoderOptions with
 * JxlEncoderOptionsSetInteger.
 * NOTE: this enum includes most but not all encoder options. The image quality
 * is a frame option that can be set with JxlEncoderOptionsSetDistance instead.
 * Options that apply globally, rather than per-frame, are set with their own
 * functions and do not use the per-frame JxlEncoderOptions.
 */
typedef enum {
  /** Sets encoder effort/speed level without affecting decoding speed. Valid
   * values are, from faster to slower speed: 1:lightning 2:thunder 3:falcon
   * 4:cheetah 5:hare 6:wombat 7:squirrel 8:kitten 9:tortoise.
   * Default: squirrel (7).
   */
  JXL_ENC_OPTION_EFFORT = 0,

  /** Sets the decoding speed tier for the provided options. Minimum is 0
   * (slowest to decode, best quality/density), and maximum is 4 (fastest to
   * decode, at the cost of some quality/density). Default is 0.
   */
  JXL_ENC_OPTION_DECODING_SPEED = 1,

  /** Sets resampling option. If enabled, the image is downsampled before
   * compression, and upsampled to original size in the decoder. Integer option,
   * use -1 for the default behavior (resampling only applied for low quality),
   * 1 for no downsampling (1x1), 2 for 2x2 downsampling, 4 for 4x4
   * downsampling, 8 for 8x8 downsampling.
   */
  JXL_ENC_OPTION_RESAMPLING = 2,

  /** Similar to JXL_ENC_OPTION_RESAMPLING, but for extra channels. Integer
   * option, use -1 for the default behavior (depends on encoder
   * implementation), 1 for no downsampling (1x1), 2 for 2x2 downsampling, 4 for
   * 4x4 downsampling, 8 for 8x8 downsampling.
   */
  JXL_ENC_OPTION_EXTRA_CHANNEL_RESAMPLING = 3,

  /** Enables or disables noise generation. Integer option, use -1 for the
   * encoder default, 0 to disable, 1 to enable.
   */
  JXL_ENC_OPTION_NOISE = 4,

  /** Enables or disables dots generation. Integer option, use -1 for the
   * encoder default, 0 to disable, 1 to enable.
   */
  JXL_ENC_OPTION_DOTS = 5,

  /** Enables or disables patches generation. Integer option, use -1 for the
   * encoder default, 0 to disable, 1 to enable.
   */
  JXL_ENC_OPTION_PATCHES = 6,

  /** Enables or disables the gaborish filter. Integer option, use -1 for the
   * encoder default, 0 to disable, 1 to enable.
   */
  JXL_ENC_OPTION_GABORISH = 7,

  /** Enables modular encoding. Integer option, use -1 for default (encoder
   * chooses), 0 to enforce VarDCT mode (e.g. for photographic images), 1 to
   * enforce modular mode (e.g. for lossless images).
   */
  JXL_ENC_OPTION_MODULAR = 8,

  /** Enum value not to be used as an option. This value is added to force the
   * C compiler to have the enum to take a known size.
   */
  JXL_ENC_OPTION_FILL_ENUM = 65535,

} JxlEncoderOptionId;

/**
 * Creates an instance of JxlEncoder and initializes it.
 *
 * @p memory_manager will be used for all the library dynamic allocations made
 * from this instance. The parameter may be NULL, in which case the default
 * allocator will be used. See jpegxl/memory_manager.h for details.
 *
 * @param memory_manager custom allocator function. It may be NULL. The memory
 *        manager will be copied internally.
 * @return @c NULL if the instance can not be allocated or initialized
 * @return pointer to initialized JxlEncoder otherwise
 */
JXL_EXPORT JxlEncoder* JxlEncoderCreate(const JxlMemoryManager* memory_manager);

/**
 * Re-initializes a JxlEncoder instance, so it can be re-used for encoding
 * another image. All state and settings are reset as if the object was
 * newly created with JxlEncoderCreate, but the memory manager is kept.
 *
 * @param enc instance to be re-initialized.
 */
JXL_EXPORT void JxlEncoderReset(JxlEncoder* enc);

/**
 * Deinitializes and frees JxlEncoder instance.
 *
 * @param enc instance to be cleaned up and deallocated.
 */
JXL_EXPORT void JxlEncoderDestroy(JxlEncoder* enc);

/**
 * Set the parallel runner for multithreading. May only be set before starting
 * encoding.
 *
 * @param enc encoder object.
 * @param parallel_runner function pointer to runner for multithreading. It may
 *        be NULL to use the default, single-threaded, runner. A multithreaded
 *        runner should be set to reach fast performance.
 * @param parallel_runner_opaque opaque pointer for parallel_runner.
 * @return JXL_ENC_SUCCESS if the runner was set, JXL_ENC_ERROR
 * otherwise (the previous runner remains set).
 */
JXL_EXPORT JxlEncoderStatus
JxlEncoderSetParallelRunner(JxlEncoder* enc, JxlParallelRunner parallel_runner,
                            void* parallel_runner_opaque);

/**
 * Encodes JPEG XL file using the available bytes. @p *avail_out indicates how
 * many output bytes are available, and @p *next_out points to the input bytes.
 * *avail_out will be decremented by the amount of bytes that have been
 * processed by the encoder and *next_out will be incremented by the same
 * amount, so *next_out will now point at the amount of *avail_out unprocessed
 * bytes.
 *
 * The returned status indicates whether the encoder needs more output bytes.
 * When the return value is not JXL_ENC_ERROR or JXL_ENC_SUCCESS, the encoding
 * requires more JxlEncoderProcessOutput calls to continue.
 *
 * @param enc encoder object.
 * @param next_out pointer to next bytes to write to.
 * @param avail_out amount of bytes available starting from *next_out.
 * @return JXL_ENC_SUCCESS when encoding finished and all events handled.
 * @return JXL_ENC_ERROR when encoding failed, e.g. invalid input.
 * @return JXL_ENC_NEED_MORE_OUTPUT more output buffer is necessary.
 */
JXL_EXPORT JxlEncoderStatus JxlEncoderProcessOutput(JxlEncoder* enc,
                                                    uint8_t** next_out,
                                                    size_t* avail_out);

/**
 * Sets the buffer to read JPEG encoded bytes from for the next frame to encode.
 *
 * If JxlEncoderSetBasicInfo has not yet been called, calling
 * JxlEncoderAddJPEGFrame will implicitly call it with the parameters of the
 * added JPEG frame.
 *
 * If JxlEncoderSetColorEncoding or JxlEncoderSetICCProfile has not yet been
 * called, calling JxlEncoderAddJPEGFrame will implicitly call it with the
 * parameters of the added JPEG frame.
 *
 * If the encoder is set to store JPEG reconstruction metadata using @ref
 * JxlEncoderStoreJPEGMetadata and a single JPEG frame is added, it will be
 * possible to losslessly reconstruct the JPEG codestream.
 *
 * @param options set of encoder options to use when encoding the frame.
 * @param buffer bytes to read JPEG from. Owned by the caller and its contents
 * are copied internally.
 * @param size size of buffer in bytes.
 * @return JXL_ENC_SUCCESS on success, JXL_ENC_ERROR on error
 */
JXL_EXPORT JxlEncoderStatus JxlEncoderAddJPEGFrame(
    const JxlEncoderOptions* options, const uint8_t* buffer, size_t size);

/**
 * Sets the buffer to read pixels from for the next image to encode. Must call
 * JxlEncoderSetBasicInfo before JxlEncoderAddImageFrame.
 *
 * Currently only some pixel formats are supported:
 * - JXL_TYPE_UINT8
 * - JXL_TYPE_UINT16
 * - JXL_TYPE_FLOAT16, with nominal range 0..1
 * - JXL_TYPE_FLOAT, with nominal range 0..1
 *
 * The color profile of the pixels depends on the value of uses_original_profile
 * in the JxlBasicInfo. If true, the pixels are assumed to be encoded in the
 * original profile that is set with JxlEncoderSetColorEncoding or
 * JxlEncoderSetICCProfile. If false, the pixels are assumed to be nonlinear
 * sRGB for integer data types (JXL_TYPE_UINT8, JXL_TYPE_UINT16), and linear
 * sRGB for floating point data types (JXL_TYPE_FLOAT16, JXL_TYPE_FLOAT).
 *
 * @param options set of encoder options to use when encoding the frame.
 * @param pixel_format format for pixels. Object owned by the caller and its
 * contents are copied internally.
 * @param buffer buffer type to input the pixel data from. Owned by the caller
 * and its contents are copied internally.
 * @param size size of buffer in bytes.
 * @return JXL_ENC_SUCCESS on success, JXL_ENC_ERROR on error
 */
JXL_EXPORT JxlEncoderStatus JxlEncoderAddImageFrame(
    const JxlEncoderOptions* options, const JxlPixelFormat* pixel_format,
    const void* buffer, size_t size);

/**
 * Declares that this encoder will not encode anything further.
 *
 * Must be called between JxlEncoderAddImageFrame/JPEGFrame of the last frame
 * and the next call to JxlEncoderProcessOutput, or JxlEncoderProcessOutput
 * won't output the last frame correctly.
 *
 * @param enc encoder object.
 */
JXL_EXPORT void JxlEncoderCloseInput(JxlEncoder* enc);

/**
 * Sets the original color encoding of the image encoded by this encoder. This
 * is an alternative to JxlEncoderSetICCProfile and only one of these two must
 * be used. This one sets the color encoding as a @ref JxlColorEncoding, while
 * the other sets it as ICC binary data.
 *
 * @param enc encoder object.
 * @param color color encoding. Object owned by the caller and its contents are
 * copied internally.
 * @return JXL_ENC_SUCCESS if the operation was successful, JXL_ENC_ERROR or
 * JXL_ENC_NOT_SUPPORTED otherwise
 */
JXL_EXPORT JxlEncoderStatus
JxlEncoderSetColorEncoding(JxlEncoder* enc, const JxlColorEncoding* color);

/**
 * Sets the original color encoding of the image encoded by this encoder as an
 * ICC color profile. This is an alternative to JxlEncoderSetColorEncoding and
 * only one of these two must be used. This one sets the color encoding as ICC
 * binary data, while the other defines it as a @ref JxlColorEncoding.
 *
 * @param enc encoder object.
 * @param icc_profile bytes of the original ICC profile
 * @param size size of the icc_profile buffer in bytes
 * @return JXL_ENC_SUCCESS if the operation was successful, JXL_ENC_ERROR or
 * JXL_ENC_NOT_SUPPORTED otherwise
 */
JXL_EXPORT JxlEncoderStatus JxlEncoderSetICCProfile(JxlEncoder* enc,
                                                    const uint8_t* icc_profile,
                                                    size_t size);

/**
 * Initializes a JxlBasicInfo struct to default values.
 * For forwards-compatibility, this function has to be called before values
 * are assigned to the struct fields.
 * The default values correspond to an 8-bit RGB image, no alpha or any
 * other extra channels.
 *
 * @param info global image metadata. Object owned by the caller.
 */
JXL_EXPORT void JxlEncoderInitBasicInfo(JxlBasicInfo* info);

/**
 * Sets the global metadata of the image encoded by this encoder.
 *
 * @param enc encoder object.
 * @param info global image metadata. Object owned by the caller and its
 * contents are copied internally.
 * @return JXL_ENC_SUCCESS if the operation was successful,
 * JXL_ENC_ERROR or JXL_ENC_NOT_SUPPORTED otherwise
 */
JXL_EXPORT JxlEncoderStatus JxlEncoderSetBasicInfo(JxlEncoder* enc,
                                                   const JxlBasicInfo* info);

/**
 * Sets a frame-specific option of integer type to the encoder options.
 * The JxlEncoderOptionId argument determines which option is set.
 *
 * @param options set of encoder options to update with the new mode.
 * @param option ID of the option to set.
 * @param value Integer value to set for this option.
 * @return JXL_ENC_SUCCESS if the operation was successful, JXL_ENC_ERROR in
 * case of an error, such as invalid or unknown option id, or invalid integer
 * value for the given option. If an error is returned, the state of the
 * JxlEncoderOptions object is still valid and is the same as before this
 * function was called.
 */
JXL_EXPORT JxlEncoderStatus JxlEncoderOptionsSetInteger(
    JxlEncoderOptions* options, JxlEncoderOptionId option, int32_t value);

/** Forces the encoder to use the box-based JPEG XL container format (BMFF).
 *
 * If enabled, the encoder always uses the container format, even if not
 * necessary. If disabled, the encoder only uses the container format if
 * required (such as for JPEG metadata @ref JxlEncoderStoreJPEGMetadata), and
 * otherwise writes a direct codestream.
 *
 * By default this setting is disabled.
 *
 * This setting can only be set at the beginning, before encoding starts.
 *
 * @param enc encoder object.
 * @param force_container true if the encoder should always output the JPEG XL
 * container format.
 * @return JXL_ENC_SUCCESS if the operation was successful, JXL_ENC_ERROR
 * otherwise.
 */
JXL_EXPORT JxlEncoderStatus JxlEncoderUseContainer(JxlEncoder* enc,
                                                   JXL_BOOL force_container);

/**
 * Configure the encoder to store JPEG reconstruction metadata in the JPEG XL
 * container.
 *
 * If this is set to true and a single JPEG frame is added, it will be
 * possible to losslessly reconstruct the JPEG codestream.
 *
 * This setting can only be set at the beginning, before encoding starts.
 *
 * @param enc encoder object.
 * @param store_jpeg_metadata true if the encoder should store JPEG metadata.
 * @return JXL_ENC_SUCCESS if the operation was successful, JXL_ENC_ERROR
 * otherwise.
 */
JXL_EXPORT JxlEncoderStatus
JxlEncoderStoreJPEGMetadata(JxlEncoder* enc, JXL_BOOL store_jpeg_metadata);

/** Sets the feature level of the JPEG XL codestream. Valid values are 5 and
 * 10.
 *
 * Level 5: for end-user image delivery, this level is the most widely
 * supported level by image decoders and the recommended level to use unless a
 * level 10 feature is absolutely necessary. Supports a maximum resolution
 * 268435456 pixels total with a maximum width or height of 262144 pixels,
 * maximum 16-bit color channel depth, maximum 120 frames per second for
 * animation, maximum ICC color profile size of 4 MiB, it allows all color
 * models and extra channel types except CMYK and the JXL_CHANNEL_BLACK extra
 * channel, and a maximum of 4 extra channels in addition to the 3 color
 * channels. It also sets boundaries to certain internally used coding tools.
 *
 * Level 10: this level removes or increases the bounds of most of the level
 * 5 limitations, allows CMYK color and up to 32 bits per color channel, but
 * may be less widely supported.
 *
 * The default value is 5. To use level 10 features, the setting must be
 * explicitly set to 10, the encoder will not automatically enable it. If
 * incompatible parameters such as too high image resolution for the current
 * level are set, the encoder will return an error. For internal coding tools,
 * the encoder will only use those compatible with the level setting.
 *
 * This setting can only be set at the beginning, before encoding starts.
 */
JXL_EXPORT JxlEncoderStatus JxlEncoderSetCodestreamLevel(JxlEncoder* enc,
                                                         int level);

/**
 * Enables lossless encoding.
 *
 * This is not an option like the others on itself, but rather while enabled it
 * overrides a set of existing options (such as distance and modular mode) that
 * enables bit-for-bit lossless encoding.
 *
 * When disabled, those options are not overridden, but since those options
 * could still have been manually set to a combination that operates losslessly,
 * using this function with lossless set to JXL_DEC_FALSE does not guarantee
 * lossy encoding, though the default set of options is lossy.
 *
 * @param options set of encoder options to update with the new mode
 * @param lossless whether to override options for lossless mode
 * @return JXL_ENC_SUCCESS if the operation was successful, JXL_ENC_ERROR
 * otherwise.
 */
JXL_EXPORT JxlEncoderStatus
JxlEncoderOptionsSetLossless(JxlEncoderOptions* options, JXL_BOOL lossless);

/**
 * @param options set of encoder options to update with the new mode.
 * @param effort the effort value to set.
 * @return JXL_ENC_SUCCESS if the operation was successful, JXL_ENC_ERROR
 * otherwise.
 *
 * DEPRECATED: use JxlEncoderOptionsSetInteger(options, JXL_ENC_OPTION_EFFORT,
 * effort)) instead.
 */
JXL_EXPORT JXL_DEPRECATED JxlEncoderStatus
JxlEncoderOptionsSetEffort(JxlEncoderOptions* options, int effort);

/**
 * @param options set of encoder options to update with the new decoding speed
 * tier.
 * @param tier the decoding speed tier to set.
 * @return JXL_ENC_SUCCESS if the operation was successful, JXL_ENC_ERROR
 * otherwise.
 *
 * DEPRECATED: use JxlEncoderOptionsSetInteger(options,
 * JXL_ENC_OPTION_DECODING_SPEED, tier)) instead.
 */
JXL_EXPORT JXL_DEPRECATED JxlEncoderStatus
JxlEncoderOptionsSetDecodingSpeed(JxlEncoderOptions* options, int tier);

/**
 * Sets the distance level for lossy compression: target max butteraugli
 * distance, lower = higher quality. Range: 0 .. 15.
 * 0.0 = mathematically lossless (however, use JxlEncoderOptionsSetLossless
 * instead to use true lossless, as setting distance to 0 alone is not the only
 * requirement). 1.0 = visually lossless. Recommended range: 0.5 .. 3.0. Default
 * value: 1.0.
 *
 * @param options set of encoder options to update with the new mode.
 * @param distance the distance value to set.
 * @return JXL_ENC_SUCCESS if the operation was successful, JXL_ENC_ERROR
 * otherwise.
 */
JXL_EXPORT JxlEncoderStatus
JxlEncoderOptionsSetDistance(JxlEncoderOptions* options, float distance);

/**
 * Create a new set of encoder options, with all values initially copied from
 * the @p source options, or set to default if @p source is NULL.
 *
 * The returned pointer is an opaque struct tied to the encoder and it will be
 * deallocated by the encoder when JxlEncoderDestroy() is called. For functions
 * taking both a @ref JxlEncoder and a @ref JxlEncoderOptions, only
 * JxlEncoderOptions created with this function for the same encoder instance
 * can be used.
 *
 * @param enc encoder object.
 * @param source source options to copy initial values from, or NULL to get
 * defaults initialized to defaults.
 * @return the opaque struct pointer identifying a new set of encoder options.
 */
JXL_EXPORT JxlEncoderOptions* JxlEncoderOptionsCreate(
    JxlEncoder* enc, const JxlEncoderOptions* source);

/**
 * Sets a color encoding to be sRGB.
 *
 * @param color_encoding color encoding instance.
 * @param is_gray whether the color encoding should be gray scale or color.
 */
JXL_EXPORT void JxlColorEncodingSetToSRGB(JxlColorEncoding* color_encoding,
                                          JXL_BOOL is_gray);

/**
 * Sets a color encoding to be linear sRGB.
 *
 * @param color_encoding color encoding instance.
 * @param is_gray whether the color encoding should be gray scale or color.
 */
JXL_EXPORT void JxlColorEncodingSetToLinearSRGB(
    JxlColorEncoding* color_encoding, JXL_BOOL is_gray);

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif

#endif /* JXL_ENCODE_H_ */

/** @}*/
