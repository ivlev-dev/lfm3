use crate::{GgufFile, MetadataValue};
use lfm_domain::MetadataSource;

impl MetadataSource for GgufFile {
    fn get_str(&self, key: &str) -> Option<&str> {
        GgufFile::get_str(self, key)
    }

    fn get_u32(&self, key: &str) -> Option<u32> {
        GgufFile::get_u32(self, key)
    }

    fn get_f32(&self, key: &str) -> Option<f32> {
        match self.metadata().get(key) {
            Some(MetadataValue::F32(v)) => Some(*v),
            Some(MetadataValue::F64(v)) => Some(*v as f32),
            Some(MetadataValue::U32(v)) => Some(*v as f32),
            Some(MetadataValue::I32(v)) => Some(*v as f32),
            _ => None,
        }
    }

    fn get_i32(&self, key: &str) -> Option<i32> {
        match self.metadata().get(key) {
            Some(MetadataValue::I32(v)) => Some(*v),
            Some(MetadataValue::U32(v)) => Some(*v as i32),
            Some(MetadataValue::I64(v)) => Some(*v as i32),
            Some(MetadataValue::U64(v)) => Some(*v as i32),
            _ => None,
        }
    }

    fn get_i32_array(&self, key: &str) -> Option<Vec<i32>> {
        GgufFile::get_i32_array(self, key)
    }

    fn get_bool_array(&self, key: &str) -> Option<Vec<bool>> {
        match self.metadata().get(key) {
            Some(MetadataValue::Array(a)) => Some(
                a.values
                    .iter()
                    .map(|v| match v {
                        MetadataValue::Bool(b) => *b,
                        MetadataValue::U32(x) => *x != 0,
                        MetadataValue::I32(x) => *x != 0,
                        _ => false,
                    })
                    .collect(),
            ),
            _ => None,
        }
    }

    fn tensor_dims(&self, name: &str) -> Option<Vec<u64>> {
        self.tensor(name).map(|t| t.dims.clone())
    }

    fn has_tensor(&self, name: &str) -> bool {
        self.tensor(name).is_some()
    }
}
