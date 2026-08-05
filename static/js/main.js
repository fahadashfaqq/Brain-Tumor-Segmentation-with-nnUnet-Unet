
function previewImage(event) {
    const reader = new FileReader();
    reader.onload = function(){
        const img = document.createElement("img");
        img.src = reader.result;
        const preview = document.querySelector(".preview");
        if (preview) preview.remove();
    }
    reader.readAsDataURL(event.target.files[0]);
}
