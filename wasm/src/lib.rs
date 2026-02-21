use wasm_bindgen::prelude::*;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen]
pub struct SubjectRefiner {
    width: usize,
    height: usize,
    // Store previous frames for temporal smoothing
    history: Vec<Vec<f32>>,
    max_history: usize,
}

#[wasm_bindgen]
impl SubjectRefiner {
    #[wasm_bindgen(constructor)]
    pub fn new(width: usize, height: usize, max_history: usize) -> SubjectRefiner {
        SubjectRefiner {
            width,
            height,
            history: Vec::new(),
            max_history,
        }
    }

    /// Process a new mask frame:
    /// 1. Temporal smoothing
    /// 2. Morphology (Erosion + Dilation)
    /// 3. Component Isolation (seeding from click)
    pub fn refine_mask(&mut self, input_mask: &[f32], click_x: f32, click_y: f32) -> Vec<f32> {
        let size = self.width * self.height;
        if input_mask.len() != size {
            return input_mask.to_vec(); // Fallback if size mismatch
        }

        // 1. Add to history and calculate temporal average
        let mut averaged_mask = vec![0.0; size];
        self.history.push(input_mask.to_vec());
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }

        let history_len = self.history.len() as f32;
        for h in &self.history {
            for i in 0..size {
                averaged_mask[i] += h[i] / history_len;
            }
        }

        // 2. Thresholding and Erosion
        // We erode to break "bridges" between touching objects
        let mut eroded = vec![0u8; size];
        let threshold = 0.5;
        for i in 0..size {
            if averaged_mask[i] > threshold {
                eroded[i] = 1;
            }
        }

        let kernel_size = 5; // Adjust based on needs
        let eroded = self.erode(&eroded, kernel_size);

        // 3. Flood Fill (Connected Component) to isolate the clicked object
        let clx = (click_x * self.width as f32) as usize;
        let cly = (click_y * self.height as f32) as usize;
        let mut isolated = vec![0u8; size];

        if clx < self.width && cly < self.height {
            self.flood_fill(&eroded, &mut isolated, clx, cly);
        } else {
            // If click is out of bounds, fallback to full eroded
            isolated = eroded.clone();
        }

        // 4. Dilation to restore edges
        let dilated = self.dilate(&isolated, kernel_size);

        // 5. Re-apply original confidence values to the isolated blob
        let mut final_mask = vec![0.0; size];
        for i in 0..size {
            if dilated[i] > 0 && input_mask[i] > 0.1 {
                // Keep the smooth edges of the original AI mask, but only within our isolated zone
                final_mask[i] = input_mask[i];
            }
        }

        final_mask
    }

    fn erode(&self, img: &[u8], radius: i32) -> Vec<u8> {
        let mut out = vec![0; img.len()];
        let w = self.width as i32;
        let h = self.height as i32;

        for y in 0..h {
            for x in 0..w {
                let mut min_val = 1;
                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        if dx * dx + dy * dy <= radius * radius {
                            let nx = x + dx;
                            let ny = y + dy;
                            if nx >= 0 && nx < w && ny >= 0 && ny < h {
                                let idx = (ny * w + nx) as usize;
                                if img[idx] == 0 {
                                    min_val = 0;
                                }
                            } else {
                                min_val = 0;
                            }
                        }
                    }
                }
                out[(y * w + x) as usize] = min_val;
            }
        }
        out
    }

    fn dilate(&self, img: &[u8], radius: i32) -> Vec<u8> {
        let mut out = vec![0; img.len()];
        let w = self.width as i32;
        let h = self.height as i32;

        for y in 0..h {
            for x in 0..w {
                if img[(y * w + x) as usize] == 1 {
                    for dy in -radius..=radius {
                        for dx in -radius..=radius {
                            if dx * dx + dy * dy <= radius * radius {
                                let nx = x + dx;
                                let ny = y + dy;
                                if nx >= 0 && nx < w && ny >= 0 && ny < h {
                                    out[(ny * w + nx) as usize] = 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        out
    }

    fn flood_fill(&self, img: &[u8], out: &mut [u8], start_x: usize, start_y: usize) {
        let w = self.width;
        let h = self.height;

        let start_idx = start_y * w + start_x;
        
        // Find nearest 1 if starting point is 0
        let mut q = std::collections::VecDeque::new();
        
        if img[start_idx] == 1 {
            q.push_back((start_x, start_y));
        } else {
            // Search nearby for a 1
            let mut found = false;
            let radius = 20;
            for r in 1..=radius {
                for dy in -r..=r {
                    for dx in -r..=r {
                        let nx = start_x as i32 + dx;
                        let ny = start_y as i32 + dy;
                        if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                            if img[(ny * w as i32 + nx) as usize] == 1 {
                                q.push_back((nx as usize, ny as usize));
                                found = true;
                                break;
                            }
                        }
                    }
                    if found { break; }
                }
                if found { break; }
            }
            if !found { return; }
        }

        while let Some((x, y)) = q.pop_front() {
            let idx = y * w + x;
            if out[idx] == 0 && img[idx] == 1 {
                out[idx] = 1;
                if x > 0 { q.push_back((x - 1, y)); }
                if x < w - 1 { q.push_back((x + 1, y)); }
                if y > 0 { q.push_back((x, y - 1)); }
                if y < h - 1 { q.push_back((x, y + 1)); }
            }
        }
    }
}
