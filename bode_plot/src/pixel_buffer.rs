use std::borrow::{Borrow, BorrowMut};

pub struct PixelBuffer(Vec<u32>);

impl PixelBuffer {
    #[inline]
    #[allow(unused)]
    pub fn new(width: usize, height: usize) -> Self {
        Self(vec![0; width * height])
    }
}
impl Borrow<[u8]> for PixelBuffer {
    fn borrow(&self) -> &[u8] {
        // Safe for alignment: align_of(u8) <= align_of(u32)
        // Safe for cast: u32 can be thought of as being transparent over [u8; 4]
        unsafe {
            std::slice::from_raw_parts(
                self.0.as_ptr() as *const u8,
                self.0.len() * 4
            )
        }
    }
}
impl BorrowMut<[u8]> for PixelBuffer {
    fn borrow_mut(&mut self) -> &mut [u8] {
        // Safe for alignment: align_of(u8) <= align_of(u32)
        // Safe for cast: u32 can be thought of as being transparent over [u8; 4]
        unsafe {
            std::slice::from_raw_parts_mut(
                self.0.as_mut_ptr() as *mut u8,
                self.0.len() * 4
            )
        }
    }
}
impl Borrow<[u32]> for PixelBuffer {
    fn borrow(&self) -> &[u32] {
        self.0.as_slice()
    }
}
impl BorrowMut<[u32]> for PixelBuffer {
    fn borrow_mut(&mut self) -> &mut [u32] {
        self.0.as_mut_slice()
    }
}