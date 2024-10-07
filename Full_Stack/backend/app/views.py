import PyPDF2
import os
from langchain.llms import OpenAI
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage


def parse_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfFileReader(file)
        num_pages = len(pdf_reader.pages)
        extracted_text = ""
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            extracted_text += page.extractText()
        return extracted_text


@csrf_exempt
def upload_pdf(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage(location=settings.MEDIA_ROOT)
        filename = fs.save(myfile.name, myfile)
        file_path = fs.path(filename)

        if os.path.isfile(file_path):
            content = parse_pdf(file_path)
            print(content)
            request.session['pdf_content'] = content
            return JsonResponse({'status': 'success', 'content': content})
        else:
            return JsonResponse({'status': 'error', 'message': 'File not found on server.'})

    return JsonResponse({'status': 'fail', 'message': 'POST method with a file is expected.'})


@csrf_exempt
def answer(request):
    if 'pdf_content' not in request.session:
        fs = FileSystemStorage(location=settings.MEDIA_ROOT)

        files = fs.listdir('')
        pdf_files = [file for file in files[1] if file.endswith('.pdf')]

        if pdf_files:
            sorted_files = sorted(
                pdf_files, key=lambda x: os.path.getmtime(fs.path(x)), reverse=True)

            latest_file_path = fs.path(sorted_files[0])

            if os.path.isfile(latest_file_path):
                content = parse_pdf(latest_file_path)
                request.session['pdf_content'] = content

    source = request.session['pdf_content']
    # subject = request.POST['subject']
    question = request.POST['question']
    api_key = os.environ["OPENAI_API_KEY"] = "API_KEY"

    prompt = source + "\nBased on the above context answer the following question, " + question
    llm = OpenAI(openai_api_key=api_key, temperature=0.9)

    answer = llm(prompt)
    return JsonResponse({'status': "success", 'message': answer})
