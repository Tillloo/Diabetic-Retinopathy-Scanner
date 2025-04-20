let previousFile = null;

function handleFileInputChange(input) {
  const file = input.files[0];

  if (file) {
    previousFile = file;

    const preview = document.getElementById("preview");
    preview.src = URL.createObjectURL(file);
    preview.style.display = "block";
  } else if (previousFile) {
    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(previousFile);
    input.files = dataTransfer.files;

    const preview = document.getElementById("preview");
    preview.src = URL.createObjectURL(previousFile);
    preview.style.display = "block";
  }
}
