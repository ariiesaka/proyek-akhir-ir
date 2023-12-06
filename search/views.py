import os.path

from django.shortcuts import render, redirect
from django.contrib import messages
from .bsbi import BSBIIndex
from .letor import Letor
from .constants import OS_SEP_WIN

BSBI_instance = BSBIIndex()
letor_instance = Letor()


def home(request):
    return render(request, 'index.html')


def search(request):
    query = request.GET.get('query')

    if query is None:
        return redirect('home')

    page = int(request.GET.get('page', 0))

    result = []
    start = page * 10
    end = start + 10

    docs = list(letor_instance.predict(query, BSBI_instance.retrieve_bm25(query, k=100)))

    if len(docs) == 0:
        messages.error(request, f"Hasil {query} tidak ditemukan", extra_tags="danger")
        return redirect('home')

    if start > len(docs):
        start = 0
        end = 10

    for i, (score, doc, content) in enumerate(docs):
        if start <= i < end:
            result.append((doc, doc.split(OS_SEP_WIN)[-1][:-4], content))

    response = {
        'query': query,
        'result': result,
        'pages': range(((len(docs) - 1) // 10) + 1),
        'curr_page': page,
    }

    return render(request, 'search.html', response)


def detail(request, doc_id):
    try:
        path_parts = doc_id.split(OS_SEP_WIN)
        with open(os.path.join('search', 'collections', path_parts[-2], path_parts[-1]), 'r') as f:
            response = {
                'title': path_parts[-1],
                'text': f.read()
            }

        return render(request, 'detail.html', response)
    except FileNotFoundError:
        messages.error(request, f"Dokumen {doc_id} tidak ditemukan", extra_tags="danger")
        return redirect('home')
