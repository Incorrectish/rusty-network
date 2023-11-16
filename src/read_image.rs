use image::io::Reader;

pub(crate) fn read_print_image(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let image = Reader::open(path)?.decode()?;
    println!("{image:?}");
    Ok(())
}
