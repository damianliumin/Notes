#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <dirent.h>
#include <ctype.h>
#include <stdlib.h>
#include <sys/types.h>

char version[] = "pstree (PSmisc) UNKNOWN\n"
"Copyright (C) 1993-2019 Werner Almesberger and Craig Small\n\n"
"PSmisc comes with ABSOLUTELY NO WARRANTY.\n"
"This is free software, and you are welcome to redistribute it under\n"
"the terms of the GNU General Public License.\n"
"For more information about these matters, see the files named COPYING.\n";

enum {CMD_P = 1, CMD_N = 2, CMD_V = 4};
static int arg_rec = 0;

typedef struct process{
  /* Process Info */
  pid_t pid;
  pid_t ppid;
  char name[256];
  /* List Struct */
  struct process* next;
  /* Tree Struct */
  struct process* child;
  struct process* bro;
} Process;

void fetch_args(int argc, char *argv[]){
  assert(argv[0]);
  for (int i = 1; i < argc; i++) {
    assert(argv[i]);
    if(!strcmp("-p", argv[i]) || !strcmp("--show-pids", argv[i]))
      arg_rec |= CMD_P;
    else if(!strcmp("-n", argv[i]) || !strcmp("--numeric-sort", argv[i]))
      arg_rec |= CMD_N;
    else if(!strcmp("-V", argv[i]) || !strcmp("--version", argv[i]))
      arg_rec |= CMD_V;
    else
      printf("Error\n");
  }
  assert(!argv[argc]);
}

int read_process_info(char* filename, Process* pPro){
  char temp[266] = "/proc/";
  FILE* fp = fopen(strcat(strcat(temp, filename), "/status") ,"r");
  assert(fp);
  while(fscanf(fp, "%s", temp)){
    if(strcmp(temp, "Name:") == 0){
      fscanf(fp, "%s", pPro->name);
    } else if(strcmp(temp, "Pid:") == 0){
      fscanf(fp, "%s", temp);
      pPro->pid = atoi(temp);
    } else if(strcmp(temp, "PPid:") == 0){
      fscanf(fp, "%s", temp);
      pPro->ppid = atoi(temp);
      break;
    }
  }
  return (pPro->pid != 2) && (pPro->ppid != 2);
}

Process* fetch_process(){
  DIR* pDir = opendir("/proc/");
  assert(pDir);
  struct dirent* pEnt = NULL;
  Process *head = 0, *cur = 0;
  while(1){
    pEnt = readdir(pDir);
    if(!pEnt) break;
    if(isdigit(pEnt->d_name[0])){
      Process* pPro = (Process*)malloc(sizeof(Process));
      if(!read_process_info(pEnt->d_name, pPro)){
        free(pPro);
        continue;
      }
      if(head == NULL)
        cur = head = pPro;
      cur->next = pPro;
      cur = pPro;
    }
  }
  cur->next = NULL;
  return head;
}

void build_tree(Process* node, Process *head){
  Process *cur = node->next;
  node->child = NULL;
  while(cur != NULL){
    if(cur->ppid == node->pid){
      if(node->child == NULL)
        node->child = cur;
      else{
        Process *child_cur = node->child;
        while(child_cur->bro != NULL)
          child_cur = child_cur->bro;
        child_cur->bro = cur;
        cur->bro = NULL;
      }
      build_tree(cur, head);
    }
    cur = cur->next;
  }
}

void print_tree(Process* node, int level){
  for(int i = 1; i <= level ;++i)
    printf("      ");
  if(arg_rec & CMD_P)
    printf("%s(%d)\n", node->name, node->pid);
  else
    printf("%s\n", node->name);
  Process *child = node->child;
  while(child != NULL){
    print_tree(child, level + 1);
    child = child->bro;
  }
}

void free_mem(Process *cur){
  if(cur == NULL)
    return;
  free_mem(cur->next);
  free(cur);
}

int main(int argc, char *argv[]) {
  fetch_args(argc, argv);

  if(arg_rec & CMD_V){
    printf("%s", version);
    return 0;
  }

  Process *root = fetch_process();
  assert(root && root->pid == 1);

  build_tree(root, root);

  print_tree(root, 0);

  free_mem(root);

  return 0;
}
