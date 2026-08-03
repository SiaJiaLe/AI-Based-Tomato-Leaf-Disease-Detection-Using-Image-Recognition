// Privacy (report §3.8): strip EXIF/GPS metadata from the captured image.
//
// Decoding an image to a <canvas> and reading its pixels discards ALL file
// metadata (EXIF timestamps, GPS coordinates, device info) — only raw pixels
// survive. We route every user image through here before it is displayed or
// fed to the model, so location/time data never enters the app. The image is
// also never uploaded and never written to storage.

// Decode a File/Blob into a metadata-free ImageBitmap (pixels only).
// imageOrientation:'from-image' honours EXIF rotation for correct display,
// while the metadata itself is dropped.
export async function decodeClean(fileOrBlob) {
  return await createImageBitmap(fileOrBlob, { imageOrientation: 'from-image' })
}
